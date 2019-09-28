#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    

    
    Mat M;
    M.create(400,400, CV_8UC(3));

    bool exit = false;

    while(!exit)
    {
        imshow("test",M);
        int key = waitKey(10);

        exit = (key == 'q');

    }

    return 0;
}
