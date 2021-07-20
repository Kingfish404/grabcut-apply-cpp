#include <iostream>
#include "opencv2/opencv.hpp"
#include "grabcut.hpp"


class CGApply
{
private:
    cv::Mat *img;
    cv::Mat *img_orign;
    cv::Mat mask;
    // mask四状态：
    // 背景GCD_BGD：0;
    // 前景GCD_FGD：1;
    // 可能的背景GCD_PR_BGD：2;
    // 可能的前景GCD_PR_FGD：3


    cv::Rect roi;

    std::string img_path;
public:
    CGApply()
    {
        img_path = "./../images/cat-s.jpg";
        img = new cv::Mat;
        img_orign = new cv::Mat;
    }

    void readImg()
    {
        *img = cv::imread(img_path, CV_64F);
        img->copyTo(*img_orign);
        mask.create(img->size(), CV_8UC1);
        mask.setTo(GC_BGD);
        showImg();
    }

    void descImg() const
    {
        std::cout << "img size:" << img->size << std::endl;
        std::cout << "mask size:" << mask.size << std::endl;
        std::cout << "mask:" << mask << std::endl;
    }

    void onMouseClick(int event, int x, int y, int flags, void *param)
    {
        Rect rect_t;
        int width = 5;
        rect_t.x = x - width;
        rect_t.y = y - width;
        rect_t.width = 2 * width;
        rect_t.height = 2 * width;
        switch (event)
        {
            case EVENT_RBUTTONDOWN:
                std::cout << "R mouse click:" << x << " " << y << std::endl;
                (*img)(rect_t).setTo(Scalar(0, 255, 0));
                mask(rect_t).setTo(GC_BGD);
                break;
            case EVENT_LBUTTONDOWN:
                std::cout << "L mouse click:" << x << " " << y << std::endl;
                (*img)(rect_t).setTo(Scalar(255, 0, 0));
                mask(rect_t).setTo(GC_FGD);
                break;
            default:
                return;
        }
        imshow("img", *img);
    }

    void runGrabCut()
    {
        Mat bgdModel, fgdModel;
        selfGrabCut(*img_orign, mask, roi, bgdModel, fgdModel, 1);
    }

    void selectReat()
    {
        img_orign->copyTo(*img);
        roi = cv::selectROI("img", *img, false);
        std::cout << "roi:" << roi << std::endl;
        mask.setTo(GC_BGD);
        mask(roi).setTo(Scalar(GC_PR_FGD));
    }


    void showImg()
    {
        cv::Mat res;
        cv::Mat res_mask;
        mask.copyTo(res_mask);

        // 3 -> b11 -> &1=1
        img_orign->copyTo(res, mask & 1);
        res_mask.setTo(255);
        res_mask.copyTo(mask, mask & 1);

        cv::imshow("mask", mask);
        cv::imshow("res", res);
    }

    void clearImg()
    {
        img_orign->copyTo(*img);
        cv::destroyWindow("mask");
        cv::destroyWindow("res");
    }
};

CGApply apply;

void on_mouse(int event, int x, int y, int flags, void *param)
{
    apply.onMouseClick(event, x, y, flags, param);
}

int main()
{
    apply.readImg();
    apply.selectReat();
    cv::setMouseCallback("img", on_mouse);

    while (true)
    {
        char c = (char) cv::waitKey(0);
        switch (c)
        {
            case 'n':
                std::cout << "enter: n" << std::endl;
                apply.runGrabCut();
                apply.showImg();
                break;
            case 'r':
                apply.clearImg();
                apply.selectReat();
                apply.showImg();
                break;
            case 's':
                apply.selectReat();
                break;
            case 'd':
                apply.descImg();
                break;
            case 'q':
            case 27:
                std::cout << "exit" << std::endl;
                destroyAllWindows();
                return 0;
            default:
                std::cout << "enter: " << c << std::endl;
                break;
        }
    }
}
