#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <functional>

#define EPS 1e-6f


void prepare_images(const std::string &path1, const std::string &path2, cv::Mat &A, cv::Mat &B) {
    cv::Mat img1 = cv::imread(path1);
    cv::Mat img2 = cv::imread(path2);

    int max_width = 1000;

    if (img1.cols > max_width) {
        double scale = static_cast<double>(max_width) / img1.cols;
        cv::resize(img1, img1, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    if (img2.cols > max_width) {
        double scale = static_cast<double>(max_width) / img2.cols;
        cv::resize(img2, img2, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    cv::resize(img2, img2, img1.size(), 0, 0, cv::INTER_LINEAR);


    img1.convertTo(A, CV_32FC3, 1.0 / 255.0);
    img2.convertTo(B, CV_32FC3, 1.0 / 255.0);
}


template<class Func>
cv::Mat blend_generic(const cv::Mat& A, const cv::Mat& B, Func func) {
    CV_Assert(A.size() == B.size() && A.type() == B.type());

    cv::Mat res(A.size(), A.type());
    int total = A.rows * A.cols * 3;

    const float* ptrA = reinterpret_cast<const float*>(A.data);
    const float* ptrB = reinterpret_cast<const float*>(B.data);
    float* ptrR = reinterpret_cast<float*>(res.data);

    for (int i = 0; i < total; i++) {
        ptrR[i] = func(ptrA[i], ptrB[i]);
    }

    return res;
}


cv::Mat blend_transparency(const cv::Mat &A, const cv::Mat &B, double alpha) {
    return blend_generic(A, B, [alpha](float a, float b){ return (1.0f - alpha) * a + alpha * b; });
}

cv::Mat blend_darken(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b){ return std::min(a, b); });
}

cv::Mat blend_lighten(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b){ return std::max(a, b); });
}

cv::Mat blend_multiply(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b){ return a * b; });
}

cv::Mat blend_screen(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b){ return 1.0f - (1 - a) * (1 - b); });
}

cv::Mat blend_add(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b){ return std::min(a + b, 1.0f); });
}

cv::Mat blend_linear_burn(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b){ return std::max(a + b - 1.0f, 0.0f); });
}

cv::Mat blend_dodge(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b) {
        float val = (a >= 1.0f - EPS) ? 1.0f : b / (1.0f - a);
        return std::clamp(val, 0.0f, 1.0f);
    });
}

cv::Mat blend_burn(const cv::Mat& A, const cv::Mat& B) {
    return blend_generic(A, B, [](float a, float b) {
        float val = (a < EPS) ? 0.0f : 1.0f - (1.0f - b) / a;
        return std::clamp(val, 0.0f, 1.0f);
    });
}

cv::Mat blend_overlay(const cv::Mat& A, const cv::Mat& B) {
    return blend_generic(A, B, [](float a, float b) {
        float val = (b <= 0.5f) ? 2.0f * a * b : 1.0f - 2.0f * (1.0f - a) * (1.0f - b);
        return std::clamp(val, 0.0f, 1.0f);
    });
}

cv::Mat blend_hard_light(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b) {
        float val = (a <= 0.5f) ? 2.0f * a * b : 1.0f - 2.0f * (1.0f - a) * (1.0f - b);
        return std::clamp(val, 0.0f, 1.0f);
    });
}

cv::Mat blend_soft_light(const cv::Mat &A, const cv::Mat &B) {
    return blend_generic(A, B, [](float a, float b) {
        float val = (a <= 0.5f) ? (2.0f * a - 1.0f) * (b - std::pow(b, 2.0f)) + b : (2.0f * a - 1.0f) * (std::sqrt(b) - b) + b;
        return std::clamp(val, 0.0f, 1.0f);
    });
}

cv::Mat blend_difference(const cv::Mat& A, const cv::Mat& B) {
    return blend_generic(A, B, [](float a, float b){ return std::abs(a - b); });
}


bool folder_exists(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        try {
            return std::filesystem::create_directory(path);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << e.what() << std::endl;
            return false;
        }
    }
    return true;
}


void run(const std::string& path1,
    const std::string& path2,
    const std::string& out_folder,
    const std::string& out_file_name,
    const std::function<cv::Mat(const cv::Mat&, const cv::Mat&)>& blend_func) {
    if (!folder_exists(out_folder)) return;

    cv::Mat A;
    cv::Mat B;

    prepare_images(path1, path2, A, B);

    cv::Mat result = blend_func(A, B);

    cv::Mat display;
    result.convertTo(display, CV_8UC3, 255.0);

    std::string abs_out_path = out_folder + "/" + out_file_name;
    cv::imwrite(abs_out_path, display);
}


int main() {
    std::string path1 = "../imgs/1.jpg";
    std::string path2 = "../imgs/2.jpg";

    std::string out_folder = "../output";
    std::string out_file_name = "transp.jpg";

    double alpha = 0.6;


    run(path1, path2, out_folder, "transparency_0.6.jpg", [alpha](const cv::Mat& A, const cv::Mat& B) -> cv::Mat {
        return blend_transparency(A, B, alpha);
    });
    run(path1, path2, out_folder, "darken.jpg", blend_darken);
    run(path1, path2, out_folder, "lighten.jpg", blend_lighten);
    run(path1, path2, out_folder, "multiply.jpg", blend_multiply);
    run(path1, path2, out_folder, "screen.jpg", blend_screen);
    run(path1, path2, out_folder, "add.jpg", blend_add);
    run(path1, path2, out_folder, "linear_burn.jpg", blend_linear_burn);
    run(path1, path2, out_folder, "dodge.jpg", blend_dodge);
    run(path1, path2, out_folder, "burn.jpg", blend_burn);
    run(path1, path2, out_folder, "overlay.jpg", blend_overlay);
    run(path1, path2, out_folder, "hard_light.jpg", blend_hard_light);
    run(path1, path2, out_folder, "soft_light.jpg", blend_soft_light);
    run(path1, path2, out_folder, "difference.jpg", blend_difference);

}


