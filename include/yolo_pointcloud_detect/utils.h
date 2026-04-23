#ifndef UTILS_H
#define UTILS_H

#include <dirent.h>
#include <random>
#include <opencv2/opencv.hpp>

static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names)
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


static inline int get_random_int(int minThres=0, int maxThres=255){
    // 获取处于某一范围内的一个随机整数
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(minThres, maxThres);

    int random_integer = distrib(gen);

    return random_integer;
}

static inline cv::Scalar get_class_color(int cls)
{
    static std::unordered_map<int, cv::Scalar> class_color_map;

    auto it = class_color_map.find(cls);
    if (it == class_color_map.end())
    {
        std::mt19937 rng(cls);
        std::uniform_int_distribution<int> dist(0, 255);

        cv::Scalar color(
            dist(rng),  // B
            dist(rng),  // G
            dist(rng)   // R
        );

        it = class_color_map.emplace(cls, color).first;
    }

    return it->second;
}

#endif  // UTILS_H
