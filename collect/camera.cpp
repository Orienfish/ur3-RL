#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>

#include "main.h"
#include "qhyccd.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

/*
 * Global Variables
 */
int num;
int ret;
int found;
int cambinx,cambiny;
unsigned int w,h,bpp,channels;
char id[32];
unsigned char *ImgData;
qhyccd_handle *camhandle;

/*
 * camera_init - Camera Initialization
 */
extern "C" void camera_init()
{
    num = 0;
    ret = QHYCCD_ERROR;
    found = 0;
    cambinx = 1;
    cambiny = 1;
    channels = 0;
    camhandle = NULL;
    ret = InitQHYCCDResource();//不进入相机私有类中,只是初始化之前定义好的变量
    num = ScanQHYCCD();//new a class for the camera you connect
    for(int i = 0;i < num;i++)
    {
        ret = GetQHYCCDId(i,id);//get camera ID from cydev struct
        if(ret == QHYCCD_SUCCESS)
        {
            printf("Connected to the first camera from the list,id is %s\n",id);
            found = 1;
        }
    }
    if(found == 1)
    {
        camhandle = OpenQHYCCD(id);//返回设备句柄
        if(camhandle != NULL)
        {
            printf("Open QHYCCD success!\n");
        }
        ret = SetQHYCCDStreamMode(camhandle,0);//设置相机为单帧模式,streammode = 0/1
        if(ret == QHYCCD_SUCCESS)
        {
            printf("SetQHYCCDStreamMode success!\n");
        }

        ret = InitQHYCCD(camhandle);//初始化相机 new for rawarray&roiarray  InitQHYCCD->InitChipRegs
        if(ret == QHYCCD_SUCCESS)
        {
            printf("Init QHYCCD success!\n");
        }
     	
        double chipw,chiph,pixelw,pixelh;
        ret = GetQHYCCDChipInfo(camhandle,&chipw,&chiph,&w,&h,&pixelw,&pixelh,&bpp);//获取相机信息
        if(ret == QHYCCD_SUCCESS)
        {
            printf("GetQHYCCDChipInfo success!\n");
            printf("CCD/CMOS chip information:\n");
            printf("Chip width               : %3f mm\n",chipw);
            printf("Chip height              : %3f mm\n",chiph);
            printf("Chip pixel width         : %3f um\n",pixelw);
            printf("Chip pixel height        : %3f um\n",pixelh);
            printf("image width              : %d\n",w);
            printf("image height             : %d\n",h);
            printf("Camera depth             : %d\n",bpp);
			//printf("Chip Max Resolution is %d x %d,depth is %d\n",w,h,bpp);
        }
        
        ret = IsQHYCCDControlAvailable(camhandle,CAM_COLOR);
        if(ret == BAYER_GB || ret == BAYER_GR || ret == BAYER_BG || ret == BAYER_RG)
        {
            printf("This is a Color Camera\n");
            SetQHYCCDDebayerOnOff(camhandle,true);
            SetQHYCCDParam(camhandle,CONTROL_WBR,64);//set camera param by definition
            SetQHYCCDParam(camhandle,CONTROL_WBG,64);
            SetQHYCCDParam(camhandle,CONTROL_WBB,64);
        }

        ret = IsQHYCCDControlAvailable(camhandle,CONTROL_USBTRAFFIC);
        if(ret == QHYCCD_SUCCESS)
        {
            ret = SetQHYCCDParam(camhandle,CONTROL_USBTRAFFIC,50);

        }
		
        ret = IsQHYCCDControlAvailable(camhandle,CONTROL_GAIN);
        if(ret == QHYCCD_SUCCESS)
        {
            ret = SetQHYCCDParam(camhandle,CONTROL_GAIN,6);

        }

        ret = IsQHYCCDControlAvailable(camhandle,CONTROL_OFFSET);
        if(ret == QHYCCD_SUCCESS)
        {
            ret = SetQHYCCDParam(camhandle,CONTROL_OFFSET,150);

        }

        ret = SetQHYCCDParam(camhandle,CONTROL_EXPOSURE,2*1000);//170000000);//设置相机参数


        ret = SetQHYCCDParam(camhandle,CONTROL_SPEED,1);
        if(ret == QHYCCD_SUCCESS)
        {
        	printf("SetQHYCCDParam CONTROL_SPEED succeed!\n");
        }
        
        ret = SetQHYCCDResolution(camhandle,0,0,w,h);//设置相机分辨率
        if(ret == QHYCCD_SUCCESS)
        {
            printf("SetQHYCCDResolution success!\n");
        }

        
        ret = SetQHYCCDBinMode(camhandle,cambinx,cambiny);//设置相机输出图像数据的模式
        if(ret == QHYCCD_SUCCESS)
        {
            printf("SetQHYCCDBinMode success!\n");
        }
  }
}

/*
 * camera_close
 */
extern "C" void camera_close()
{
    if(camhandle)
    {
        ret = CancelQHYCCDExposingAndReadout(camhandle);//停止相机曝光和数据读取
        if(ret == QHYCCD_SUCCESS)
        {
            printf("CancelQHYCCDExposingAndReadout success!\n");
        }

            
        ret = CloseQHYCCD(camhandle);//关闭相机
        if(ret == QHYCCD_SUCCESS)
        {
            printf("Close QHYCCD success!\n");
        }

    }

    ret = ReleaseQHYCCDResource();//释放相机资源
    if(ret == QHYCCD_SUCCESS)
    {
        printf("Rlease SDK Resource  success!\n");
    }

    printf("QHYCCD | SingleFrameSample.cpp | end\n");
}

/*
 * camera_take_pic - take a picture and save it
 */
extern "C" void camera_take_pic(char * pic_name)
{
    /* take a pic */
    uint32_t length = GetQHYCCDMemLength(camhandle);//获取相机内存长度
    if(length > 0)
    {
        ImgData = (unsigned char *)malloc(length);
        memset(ImgData,0,length);
        // printf("QHYCCD | SingleFrameSample | camera length = %d\n",length);
    }

    //  for(int k=0;k<100;k++){
    ret = ExpQHYCCDSingleFrame(camhandle);//开始曝光一帧图像
    if(ret != QHYCCD_ERROR )
    {
        // printf("ExpQHYCCDSingleFrame success!\n");
        if(ret != QHYCCD_READ_DIRECTLY)
        {
               // sleep(1);
        } 
    }

        
    ret = GetQHYCCDSingleFrame(camhandle,&w,&h,&bpp,&channels,ImgData);//获取一帧图像数据
    if(ret == QHYCCD_SUCCESS && ret == QHYCCD_SUCCESS)
    {
        // printf("GetQHYCCDSingleFrame succeess! bpp = %d channels = %d\n",bpp,channels);

        IplImage *image = cvCreateImage(cvSize(w,h),bpp,channels);
        image->imageData = (char *)ImgData;
        cvNamedWindow("qhdccy", 0);
        //cvShowImage("qhdccy", image);
        cvSaveImage(pic_name, image);
        //cvWaitKey(0);
        cvDestroyWindow("qhdccy");
        cvReleaseImage(&image);
    }
}
