#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <unordered_map>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "yolo_pointcloud_detect/utils.h"
#include "yolo_pointcloud_detect/infer.h"
#include "yolo_pointcloud_detect/projector_lidar.hpp"
#include "yolo_pointcloud_detect/Lidar_parser_base.h"
#include "yolo_pointcloud_detect/types.h"
#include "yolo_pointcloud_detect/obstacle_detector.hpp"
#include "yolo_pointcloud_detect/patchworkplusplus.hpp"
#include <memory>
#include <mutex>

#include <sstream>
#include <iostream>
#include <cmath> 
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/surface/convex_hull.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <numeric>
#include <pcl/filters/voxel_grid.h>

using namespace std;

class DetectNode {
public:
    DetectNode(ros::NodeHandle& nh)
    {
        // params
        pkg_loc_ = ros::package::getPath("yolo_pointcloud_detect");
        nh.param<std::string>("trt_file", trtFile_, pkg_loc_ + std::string("/model/yolo11.plan"));
        nh.param<float>("roi_x_max", roi_x_max_, 100);
        nh.param<float>("roi_x_min", roi_x_min_, 0);
        nh.param<float>("roi_y_max", roi_y_max_, 40);
        nh.param<float>("roi_y_min", roi_y_min_, -40);
        nh.param<float>("roi_z_max", roi_z_max_, 3);
        nh.param<float>("roi_z_min", roi_z_min_, -3);
        ROI_MAX_POINT_ = Eigen::Vector4f(roi_x_max_, roi_y_max_, roi_z_max_, 1);
        ROI_MIN_POINT_ = Eigen::Vector4f(roi_x_min_, roi_y_min_, roi_z_min_, 1);
        nh.param<float>("voxel_grid_size", voxel_size_, 0.2);
        nh.param<float>("clustering/cluster_tolerance", cluster_tolerance_, 0.5);
        nh.param<int>("clustering/min_cluster_size", min_cluster_size_, 10);
        nh.param<int>("clustering/max_cluster_size", max_cluster_size_, 10000);
        nh.param<bool>("clustering/use_pca_box", use_pca_box_, true);
        nh.param<bool>("clustering/use_tracking", use_tracking_, true);
        nh.param<float>("clustering/displacement_thresh", displacement_thresh_, 1.0);
        nh.param<float>("clustering/iou_thresh", iou_thresh_, 1.0);

        // create detector (uses defaults for other constructor args)
        detector_ = std::make_unique<YoloDetector>(trtFile_);
        // Create point processor
        obstacle_detector_ = std::make_shared<lidar_obstacle_detector::ObstacleDetector<pcl::PointXYZ>>();
        obstacle_id_ = 0;
        PatchworkppGroundSeg_.reset(new PatchWorkpp<pcl::PointXYZ>(&nh));

        ROS_INFO("DetectorNode started. Engine: %s", trtFile_.c_str());

        nh.param<vector<double>>("camera/camera_matrix", camera_matrix_,vector<double>());
        nh.param<vector<double>>("camera/RT", calib_,vector<double>());
        nh.param<vector<double>>("camera/distort", distort_,vector<double>());
    
        RT_ = cv::Mat(4,4,cv::DataType<double>::type); //  旋转矩阵和平移向量
        D_ = cv::Mat(5,1,cv::DataType<double>::type); 
        K_ = cv::Mat(3,3,cv::DataType<double>::type);
        loadCalibrationData();
        ROS_INFO("CalibrationData initialized.");

        nh.param<std::string>("sensor_fusion/lidar/lidar_input_topic", lidar_sub_topic_,"/rslidar_points");//接收激光雷达话题
        nh.param<std::string>("sensor_fusion/lidar/lidar_output_topic", lidar_pub_topic_,"/fusion/rslidar_points");//发布融合后点云
        nh.param<std::string>("sensor_fusion/cam/cam_input_topic", cam_sub_topic_,"/camera/image");//接收图像话题
        nh.param<std::string>("sensor_fusion/cam/cam_output_topic", cam_pub_topic_,"/fusion/image");//发布融合后图像话题

        image_sub_ .subscribe(nh, cam_sub_topic_, 10);
        cloud_sub_ .subscribe(nh, lidar_sub_topic_, 10);
        lidar_pub_ = nh.advertise<sensor_msgs::PointCloud2>(lidar_pub_topic_, 1);
        cam_pub_ = nh.advertise<sensor_msgs::Image>(cam_pub_topic_, 1);  
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), image_sub_, cloud_sub_));
        sync_->registerCallback(
            boost::bind(&DetectNode::callback, this, _1, _2)
        );

        marker_array_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/fusion/convex_hull", 1);
        ROS_INFO("SyncSubscriber and Publisher initialized.");
    }

private:

    void loadCalibrationData(void )
    {
        // 加载 RT_ 矩阵
        RT_.at<double>(0, 0) = calib_[0]; RT_.at<double>(0, 1) = calib_[1]; RT_.at<double>(0, 2) = calib_[2]; RT_.at<double>(0, 3) = calib_[3];
        RT_.at<double>(1, 0) = calib_[4]; RT_.at<double>(1, 1) = calib_[5]; RT_.at<double>(1, 2) = calib_[6]; RT_.at<double>(1, 3) = calib_[7];
        RT_.at<double>(2, 0) = calib_[8]; RT_.at<double>(2, 1) = calib_[9]; RT_.at<double>(2, 2) = calib_[10]; RT_.at<double>(2, 3) = calib_[11];
        RT_.at<double>(3, 0) = calib_[12]; RT_.at<double>(3, 1) = calib_[13]; RT_.at<double>(3, 2) = calib_[14]; RT_.at<double>(3, 3) = calib_[15];
        // 加载相机内参矩阵
        K_.at<double>(0, 0) = camera_matrix_[0]; K_.at<double>(0, 1) = camera_matrix_[1]; K_.at<double>(0, 2) = camera_matrix_[2];
        K_.at<double>(1, 0) = camera_matrix_[3]; K_.at<double>(1, 1) = camera_matrix_[4]; K_.at<double>(1, 2) = camera_matrix_[5];
        K_.at<double>(2, 0) = camera_matrix_[6]; K_.at<double>(2, 1) = camera_matrix_[7]; K_.at<double>(2, 2) = camera_matrix_[8];
        // 加载畸变系数
        D_.at<double>(0, 0) = distort_[0]; D_.at<double>(1, 0) = distort_[1]; D_.at<double>(2, 0) = distort_[2];
        D_.at<double>(3, 0) = distort_[3]; D_.at<double>(4, 0) = distort_[4];
        // **将矩阵类型转换为 CV_32F**
        RT_.convertTo(RT_, CV_32F);
        K_.convertTo(K_, CV_32F);
        D_.convertTo(D_, CV_32F);
    }

    void callback(const sensor_msgs::ImageConstPtr& img_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        ROS_INFO("Received synchronized messages!");
        ROS_INFO("Image timestamp: %f", img_msg->header.stamp.toSec());
        ROS_INFO("Cloud timestamp: %f", cloud_msg->header.stamp.toSec());
   
        // 这里可以进行图像 + 点云融合处理
        // convert to cv::Mat BGR
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat img = cv_ptr->image;
        if (img.empty()) return;

        cv::Mat I = cv::Mat::eye(3, 3, CV_32FC1);
        cv::Mat mapX, mapY;
        cv::Mat outImg = cv::Mat(img.size(), CV_32FC3);
        cv::initUndistortRectifyMap(K_, D_, I, K_, img.size(), CV_32FC1, mapX, mapY);
        cv::remap(img, outImg, mapX, mapY, cv::INTER_LINEAR);

        // run inference (protect detector if needed)
        std::vector<Detection> res;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            std::cout<<"Running inference..."<<std::endl;
            res = detector_->inference(outImg);
            std::cout<<"Inference done. Detected "<<res.size()<<" objects."<<std::endl;
            detector_->draw_image(outImg, res, false);
        }

        // 点云下采样

        pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *current_pc_ptr);
        auto filtered_cloud = obstacle_detector_->filterCloud(current_pc_ptr, voxel_size_, ROI_MIN_POINT_, ROI_MAX_POINT_);
        auto segmented_clouds = PatchworkppGroundSeg_->estimate_ground(*filtered_cloud);

        projector_.loadPointCloud(segmented_clouds.first); // 加载点云
        projector_.setPointSize(3); // 设置点大小
        //projector_.setDisplayMode(false); // 使用距离作为颜色
        projector_.setFilterMode(true); // 启用重叠点过滤

        // 投影点云到图像
        auto projected_result = projector_.ProjectToRawMat(outImg, K_, D_, RT_(cv::Rect(0, 0, 3, 3)), RT_(cv::Rect(3, 0, 1, 3)), res);
        
        
        // 发布融合后的点云
        projected_result.first->width = 1;
        projected_result.first->height = projected_result.first->points.size();
        pcl::toROSMsg(*projected_result.first, fusion_msg_);
        fusion_msg_.header = cloud_msg->header;
        lidar_pub_.publish(fusion_msg_);

        // 3. 【关键】创建一个临时的 PointXYZ 点云，把你的自定义点拷贝进去
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (projected_result.first->empty()) {
            ROS_WARN("Projected point cloud is empty!");
            image_msg_ = cv_bridge::CvImage(img_msg->header, "bgr8", projected_result.second).toImageMsg();
            cam_pub_.publish(image_msg_);
            return;
        }
        temp_cloud->reserve(projected_result.first->size());  // 预分配，加速

        for (const auto& pt : *projected_result.first) {
            pcl::PointXYZ p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            temp_cloud->push_back(p);
        }
        // Cluster objects
        auto cloud_clusters = obstacle_detector_->clustering(temp_cloud, cluster_tolerance_, min_cluster_size_, max_cluster_size_);
        publishDetectedObjects(std::move(cloud_clusters), cloud_msg->header, projected_result.second, projected_result.first);
        // 发布融合后的图像
        image_msg_ = cv_bridge::CvImage(img_msg->header, "bgr8", projected_result.second).toImageMsg();
        cam_pub_.publish(image_msg_);
       
    }

    void publishDetectedObjects(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>&& cloud_clusters, const std_msgs::Header& header, cv::Mat& image, const pcl::PointCloud<PointXYZRGBIL>::Ptr& projected_result)
    {
        for (size_t i = 0; i < cloud_clusters.size(); ++i)
        {
            auto& cluster = cloud_clusters[i];
 
            // 获得label
            int label = getClusterMainLabel(cluster, projected_result);
            if(label >= 8) continue; // 如果没有有效的label，跳过这个cluster
            // Create Bounding Box
            Box box = use_pca_box_ ? 
                obstacle_detector_->pcaBoundingBox(cluster, obstacle_id_) : 
                obstacle_detector_->axisAlignedBoundingBox(cluster, obstacle_id_);
            
            // 为 Box 赋值凸包
            std::vector<geometry_msgs::Point32> convex_hull_points = calculateBoxVertices(box.position, box.dimension, box.quaternion);

            // 2. 投影到图像
            auto image_points = project3DBoxToImage(convex_hull_points);

            // 3. 在图像上画框
            draw3DBoxOnImage(image, image_points, label, 2);

            // 通过新的构造函数创建 Box，并赋值 convex_hull
            box = Box(obstacle_id_, box.position, box.dimension, box.quaternion);

            obstacle_id_ = (obstacle_id_ < SIZE_MAX) ? ++obstacle_id_ : 0;
            curr_boxes_.emplace_back(box);
        }
        // Re-assign Box ids based on tracking result
        if (use_tracking_)
            obstacle_detector_->obstacleTracking(prev_boxes_, curr_boxes_, displacement_thresh_, iou_thresh_);
        
        // Lookup for frame transform between the lidar frame and the target frame
        auto bbox_header = header;
        // Construct Bounding Boxes from the clusters
        visualization_msgs::MarkerArray jsk_bboxes;
        // Transform boxes from lidar frame to base_link frame, and convert to jsk and autoware msg formats
        for (auto& box : curr_boxes_)
        {
            geometry_msgs::Pose pose;
            pose.position.x = box.position(0);
            pose.position.y = box.position(1);
            pose.position.z = box.position(2);
            pose.orientation.w = box.quaternion.w();
            pose.orientation.x = box.quaternion.x();
            pose.orientation.y = box.quaternion.y();
            pose.orientation.z = box.quaternion.z();

            jsk_bboxes.markers.emplace_back(transformMarker(box, bbox_header, pose));
            //jsk_bboxes.boxes.emplace_back(transformJskBbox(box, bbox_header, pose_transformed));
        }
        marker_array_pub_.publish(std::move(jsk_bboxes));
        //pub_jsk_bboxes.publish(std::move(jsk_bboxes));
        // Update previous bounding boxes
        prev_boxes_.swap(curr_boxes_);
        curr_boxes_.clear();
    }

    // 计算Box的8个顶点
    std::vector<geometry_msgs::Point32> calculateBoxVertices(const Eigen::Vector3f& position, 
                                            const Eigen::Vector3f& dimension, 
                                            const Eigen::Quaternionf& quaternion)
    {
        // Box的长宽高（深度、宽度、高度）
        float length = dimension[0];  // 深度
        float width = dimension[1];   // 宽度
        float height = dimension[2];  // 高度

        // 定义盒子在局部坐标系中的8个顶点的相对位置
        Eigen::Vector3f local_corners[8] = {
            // 底面四个点
            Eigen::Vector3f(-length / 2, -width / 2, -height / 2),
            Eigen::Vector3f(-length / 2,  width / 2, -height / 2),
            Eigen::Vector3f( length / 2,  width / 2, -height / 2),
            Eigen::Vector3f( length / 2, -width / 2, -height / 2),
            // 顶面四个点
            Eigen::Vector3f(-length / 2, -width / 2,  height / 2),
            Eigen::Vector3f(-length / 2,  width / 2,  height / 2),
            Eigen::Vector3f( length / 2,  width / 2,  height / 2),
            Eigen::Vector3f( length / 2, -width / 2,  height / 2)
        };

        // 结果顶点存储
        std::vector<geometry_msgs::Point32> global_corners;

        // 旋转 + 平移
        for (int i = 0; i < 8; ++i)
        {
            // 旋转顶点
            Eigen::Vector3f rotated_corner = quaternion * local_corners[i];
            
            // 将旋转后的顶点平移到全局坐标系
            Eigen::Vector3f global_corner = rotated_corner + position;

            geometry_msgs::Point32 corner;

            corner.x = global_corner[0];
            corner.y = global_corner[1];
            corner.z = global_corner[2];

            // 存储结果
            global_corners.push_back(corner);
        }

        return global_corners;
    }

    visualization_msgs::Marker transformMarker(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed)
    {
        visualization_msgs::Marker marker_bbox;
        marker_bbox.header = header;
        marker_bbox.pose = pose_transformed;
        marker_bbox.type = visualization_msgs::Marker::LINE_STRIP;
        marker_bbox.id = box.id;
        geometry_msgs::Point pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8;
        pos1.x = box.dimension(0) / 2;
        pos1.y = box.dimension(1) / 2;
        pos1.z = box.dimension(2) / 2;

        pos2.x = box.dimension(0) / 2;
        pos2.y = box.dimension(1) / 2;
        pos2.z = -box.dimension(2) / 2;

        pos3.x = box.dimension(0) / 2;
        pos3.y = -box.dimension(1) / 2;
        pos3.z = -box.dimension(2) / 2;

        pos4.x = box.dimension(0) / 2;
        pos4.y = -box.dimension(1) / 2;
        pos4.z = box.dimension(2) / 2;

        pos5.x = -box.dimension(0) / 2;
        pos5.y = -box.dimension(1) / 2;
        pos5.z = box.dimension(2) / 2;

        pos6.x = -box.dimension(0) / 2;
        pos6.y = -box.dimension(1) / 2;
        pos6.z = -box.dimension(2) / 2;

        pos7.x = -box.dimension(0) / 2;
        pos7.y = box.dimension(1) / 2;
        pos7.z = -box.dimension(2) / 2;

        pos8.x = -box.dimension(0) / 2;
        pos8.y = box.dimension(1) / 2;
        pos8.z = box.dimension(2) / 2;
        marker_bbox.points.push_back(pos1);
        marker_bbox.points.push_back(pos2);
        marker_bbox.points.push_back(pos3);
        marker_bbox.points.push_back(pos4);
        marker_bbox.points.push_back(pos5);
        marker_bbox.points.push_back(pos6);
        marker_bbox.points.push_back(pos7);
        marker_bbox.points.push_back(pos8);
        marker_bbox.points.push_back(pos1);
        marker_bbox.points.push_back(pos4);
        marker_bbox.points.push_back(pos3);
        marker_bbox.points.push_back(pos6);
        marker_bbox.points.push_back(pos5);
        marker_bbox.points.push_back(pos8);
        marker_bbox.points.push_back(pos7);
        marker_bbox.points.push_back(pos2);
        marker_bbox.color.r = 1.0;
        marker_bbox.color.g = 0.0;
        marker_bbox.color.b = 0.0;
        marker_bbox.scale.x = 0.1;
        marker_bbox.color.a = 1.0;
        marker_bbox.lifetime.fromSec(0.2);
        return std::move(marker_bbox);
    };

   std::vector<cv::Point2f> project3DBoxToImage(const std::vector<geometry_msgs::Point32>& box_vertices)
    {
        std::vector<cv::Point2f> image_points;
        image_points.reserve(8);

        // 直接用你的内参 K_
        float fx = K_.at<float>(0, 0);
        float fy = K_.at<float>(1, 1);
        float cx = K_.at<float>(0, 2);
        float cy = K_.at<float>(1, 2);

        for (const auto& pt3d : box_vertices)
        {
            // 1. 3D世界点
            Eigen::Vector3f p_world(pt3d.x, pt3d.y, pt3d.z);
            cv::Mat pt_world = (cv::Mat_<float>(4, 1) << p_world.x(), p_world.y(), p_world.z(), 1.0f);

            // 2. 世界 → 相机坐标系（用你的 RT_）
            cv::Mat pt_cam = RT_ * pt_world;
            float X = pt_cam.at<float>(0, 0);
            float Y = pt_cam.at<float>(1, 0);
            float Z = pt_cam.at<float>(2, 0);

            // 过滤相机后方点
            if (Z < 1e-6)
            {
                image_points.emplace_back(-1, -1);
                continue;
            }

            // 3. 【无畸变】直接投影！
            float u = fx * X / Z + cx;
            float v = fy * Y / Z + cy;

            image_points.emplace_back(u, v);
        }

        return image_points;
    }

    void draw3DBoxOnImage(
        cv::Mat& image,
        const std::vector<cv::Point2f>& image_points,
        int label,
        int thickness = 2)
    {
        if (image_points.size() != 8)
            return;

        // ====================== 1. 根据 label 生成颜色 ======================
        static std::vector<cv::Scalar> color_map = {
            cv::Scalar(0, 255, 0),      // 绿
            cv::Scalar(255, 0, 0),      // 蓝
            cv::Scalar(0, 0, 255),      // 红
            cv::Scalar(255, 255, 0),    // 青
            cv::Scalar(255, 0, 255),    // 紫
            cv::Scalar(0, 255, 255),    // 黄
            cv::Scalar(128, 0, 0),
            cv::Scalar(0, 128, 0),
            cv::Scalar(0, 0, 128),
            cv::Scalar(128, 128, 0),
            cv::Scalar(128, 0, 128),
            cv::Scalar(0, 128, 128),
            cv::Scalar(255, 128, 0),
            cv::Scalar(255, 0, 128),
            cv::Scalar(128, 255, 0),
            cv::Scalar(0, 255, 128),
            cv::Scalar(128, 0, 255),
            cv::Scalar(0, 128, 255),
            cv::Scalar(255, 255, 128),
            cv::Scalar(255, 128, 255)
        };

        cv::Scalar color = color_map[label % color_map.size()];

        // ====================== 2. 绘制 3D 框 12 条边 ======================
        const int edges[12][2] = {
            {0,1}, {1,2}, {2,3}, {3,0},  // 底面
            {4,5}, {5,6}, {6,7}, {7,4},  // 顶面
            {0,4}, {1,5}, {2,6}, {3,7}   // 侧面
        };

        for (const auto& edge : edges)
        {
            const cv::Point2f& p1 = image_points[edge[0]];
            const cv::Point2f& p2 = image_points[edge[1]];
            if (p1.x > 0 && p1.y > 0 && p2.x > 0 && p2.y > 0)
            {
                cv::line(image, p1, p2, color, thickness);
            }
        }

        // ===================== 【超大醒目文字】 =====================
        std::string className = "unknown";
        if (label >= 0 && label < vClassNames.size()) {
            className = vClassNames[label];
        }
        std::string text = className + " (" + std::to_string(label) + ")";

        cv::Point2f pos = image_points[0];
        if (pos.x <= 0 || pos.y <= 0) return;

        // 文字上移，避免贴在框上
        pos.y -= 25;

        // ========== 核心：超大字体、超粗描边、高亮 ==========
        double fontScale = 1.2;      // 【字体大小】 1.2 = 超大
        int fontThick = 3;           // 【字体粗细】 加粗
        int outlineThick = 6;        // 【描边粗细】 超粗

        // 黑色描边（背景）
        cv::putText(image, text, pos, cv::FONT_HERSHEY_DUPLEX,
                    fontScale, cv::Scalar(255,255,255), outlineThick);
        // 白色高亮字（前景）
        cv::putText(image, text, pos, cv::FONT_HERSHEY_DUPLEX,
                    fontScale, cv::Scalar(0,0,0), fontThick);
    }


    int getClusterMainLabel(const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cluster, const pcl::PointCloud<PointXYZRGBIL>::Ptr& projected_result)
    {
        // 空点云直接返回 -1 标记无效
        if (cluster->empty()) {
            return -1;
        }

        // 构建坐标 -> label 映射表
        std::unordered_map<XYZKey, int, XYZKeyHash> xyz_to_label;
        xyz_to_label.reserve(projected_result->size());

        for (const auto& p : *projected_result) {
            xyz_to_label[{p.x, p.y, p.z}] = p.label;
        }

        // 统计 label
        std::unordered_map<int, int> label_count;
        for (const auto& p : *cluster) {
            XYZKey key{p.x, p.y, p.z};
            auto it = xyz_to_label.find(key);
            if (it != xyz_to_label.end())
                label_count[it->second]++;
        }

        // 找众数
        int max_label = -1;
        int max_count = 0;
        for (const auto& pair : label_count) {
            if (pair.second > max_count) {
                max_count = pair.second;
                max_label = pair.first;
            }
        }

        return max_label;
    }

    // ====================== 修复核心：自定义 XYZ 键值 ======================
    struct XYZKey {
        float x, y, z;

        // 必须实现 == 运算符，让 unordered_map 可以比较
        bool operator==(const XYZKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    // 哈希函数
    struct XYZKeyHash {
        size_t operator()(const XYZKey& k) const {
            size_t h1 = std::hash<float>{}(k.x);
            size_t h2 = std::hash<float>{}(k.y);
            size_t h3 = std::hash<float>{}(k.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };


    // message filter subscribers
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;

    // 定义同步策略
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image,
        sensor_msgs::PointCloud2
    > SyncPolicy;

    ros::Publisher lidar_pub_;//融合后点云可视化结果
    ros::Publisher cam_pub_;//融合后图像可视化话题
    ros::Publisher marker_array_pub_; // 发布MarkerArray消息

    // synchronizer
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    std::string lidar_sub_topic_;//接收激光雷达话题
    std::string lidar_pub_topic_;//发布融合后点云
    std::string cam_sub_topic_;//接收图像话题
    std::string cam_pub_topic_; //发布融合后图像话题

    std::string pkg_loc_;
    std::string trtFile_;

    std::unique_ptr<YoloDetector> detector_;
    std::mutex mtx_;

    cv::Mat RT_; // rotation matrix and translation vector 旋转矩阵和平移向量AUTOWARE格式，这个格式对应一般的外参标定结果（旋转平移矩阵）需要参考https://blog.csdn.net/qq_22059843/article/details/103022451
    cv::Mat D_; 
    cv::Mat K_; 
    std::vector<double> camera_matrix_;
    std::vector<double> calib_;
    std::vector<double> distort_;

    // 使用 Projector 类进行投影
    Projector projector_;

    sensor_msgs::ImagePtr image_msg_;
    sensor_msgs::PointCloud2 fusion_msg_;  //等待发送的点云消息

    std::shared_ptr<lidar_obstacle_detector::ObstacleDetector<pcl::PointXYZ>> obstacle_detector_;
    boost::shared_ptr<PatchWorkpp<pcl::PointXYZ>> PatchworkppGroundSeg_;
    size_t obstacle_id_;
    std::vector<Box> prev_boxes_, curr_boxes_;
    float roi_x_min_, roi_x_max_, roi_y_min_, roi_y_max_, roi_z_min_, roi_z_max_;
    Eigen::Vector4f ROI_MAX_POINT_;
    Eigen::Vector4f ROI_MIN_POINT_;
    float voxel_size_;
    float cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    bool use_pca_box_;
    bool use_tracking_;
    float displacement_thresh_;
    float iou_thresh_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "yolo_pointcloud_detect_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    DetectNode Detect(nh);

    ros::spin();
    return 0;
}