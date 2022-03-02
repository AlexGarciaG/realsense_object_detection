
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//Dynamic configure
#include <dynamic_reconfigure/server.h>
#include <object_detection/object_detectionConfig.h>
//Threding
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

#include "utils.h"

using namespace std;
using namespace cv;
using namespace tbb;

static const std::string OPENCV_WINDOW = "Image window";






class Erosion16 {
	private:
		cv::Mat &image;
    cv::Mat &imageTemp;
    int maskR;

	public:
		Erosion16(cv::Mat &img,cv::Mat &imgTemp, int mask) :image(img), imageTemp(imgTemp),maskR(mask){}

		Erosion16(Erosion16 &x, tbb::split): image(x.image),imageTemp(x.imageTemp),maskR(x.maskR) {}
	void operator() (const blocked_range<int> &r)const {
       /***********************************
       * Erosion16:
       * Erosion for  depth image
       * 16 bits 1 canal
       * Add black pixels when pixels are near
       * *********************************/

      //Image erosion16
      
      bool dark;
			for (int i = r.begin(); i != r.end(); i++) {
				int c = i / image.rows;
				int r = i % image.rows;
        dark = false;
        //Get mask
        for (int tempi=r-maskR; tempi<(r+maskR);tempi++){
          if((tempi>0) && (tempi <image.rows)){
            for (int tempj=c-maskR; tempj<(c+maskR);tempj++){
              if((tempj>0) && (tempj <image.cols)){
                //If found a drak pixel enable flag
                if( image.at<unsigned short>(tempi, tempj) == 0){
                  dark=true;
                }
              }else{
                  dark=true;
              }
            }
          }else{
            dark=true;
          }
        }
        //If drak pixel eflag is nable then place dark pixel otherwise copy original image pixel
        if( dark){
            imageTemp.at<unsigned short>(r, c) = 0;
          }else{
            imageTemp.at<unsigned short>(r, c) = image.at<unsigned short>(r, c) ;
        }
			}
      
		}

};

class Dilation16 {
	private:
		cv::Mat &image;
    cv::Mat &imageTemp;
    int maskR;

	public:
		Dilation16(cv::Mat &img,cv::Mat &imgTemp, int mask) :image(img), imageTemp(imgTemp),maskR(mask){}

		Dilation16(Dilation16 &x, tbb::split): image(x.image),imageTemp(x.imageTemp),maskR(x.maskR) {}
	void operator() (const blocked_range<int> &r)const {
    /***********************************
     * Dilation16:
     * Dilation for  depth image
     * 16 bits 1 canal
     * Eliminate black pixels when pixels are near
     * *********************************/
			for (int i = r.begin(); i != r.end(); i++) {
				int c = i / image.rows;
				int r = i % image.rows;
          //If the pixel is dark calculate the substitute pixel otherwise place the original
          if (image.at<unsigned short>(r, c)==0){
            //Temporal variables to get average
            unsigned long  int count = 0;
            unsigned long  int sum= 0;
            //Get mask
            for (int tempi=r-maskR; tempi<(r+maskR);tempi++){
              if((tempi>0) && (tempi <image.rows)){
                for (int tempj=c-maskR; tempj<(c+maskR);tempj++){
                  if((tempj>0) && (tempj <image.cols)){
                    count++;
                    sum= sum +image.at<unsigned short>(tempi, tempj); 
                  }else{
                    count++;
                  }
                }
              }
              else{
                count++;
              }
            }
            //place the average
            imageTemp.at<unsigned short>(r, c) = sum/count;
          }else{
            imageTemp.at<unsigned short>(r, c) = image.at<unsigned short>(r, c);
          }
			}
		}
};

class Edge16 {
	private:
		cv::Mat &image;
    cv::Mat &imageEdge;
    int varThreshold16;

	public:
		Edge16(cv::Mat &img,cv::Mat &imgEdge, int threshold16) :image(img), imageEdge(imgEdge),varThreshold16(threshold16){}

		Edge16(Edge16 &x, tbb::split): image(x.image),imageEdge(x.imageEdge),varThreshold16(x.varThreshold16) {}
	void operator() (const blocked_range<int> &r)const {
      /***********************************
     * Edge16:
     * Edge for  depth image
     * 16 bits 1 canal
     * Get the edge of the edge images
     * *********************************/
      //Define the edge image
      //Define the mask 
			int maskV[3][3] = {{-1, 0, 1},
						{-2, 0, 2},
						{-1, 0, 1}};
			int maskH[3][3] = {{ 1, 2, 1},
						{ 0, 0, 0},
						{-1,-2,-1}};	
			//Define comulative variables
			long int comulativeV;
			long int comulativeH;
			long int comulative;
			unsigned short pixelValue;

      for (int i = r.begin(); i != r.end(); i++) {
				int c = i / image.rows;
				int r = i % image.rows;
          //reset comulatives
					comulativeV =0;
					comulativeH =0;
          comulative  = 0;
					//Check if is not the edge of the picture
					if(!((r==0 || r==(image.rows-1))||(c==0 || c==(image.cols-1)))){
            //Calculate the comulative edge
						for(int tent1 = 0; tent1 <3; tent1++) {
							for(int tent2 = 0; tent2 <3; tent2++) {
								pixelValue = image.at<unsigned short>(tent1+r-1,tent2+c-1);
								//Vertical
								comulativeV =comulativeV+ pixelValue*maskV[tent1][tent2];
								//Horizontal 
								comulativeH =comulativeH+ pixelValue*maskH[tent1][tent2];
							}
						}
					}
					//Limite the Vertical comulative to 16 bit size
					
					if (comulativeV<0){
						comulativeV = 0;
					}else if (comulativeV>65535){
						comulativeV = 65535;
					}
					//Limite the Horizontal comulative to 16 bit size

					if (comulativeH<0){
						comulativeH = 0;
					}else if (comulativeH>65535){
						comulativeH = 65535;
					}
					//Combinacion
          //Calculate the Combinacion of Vertical and Horizontal comulative of 16 bit size
					comulative = sqrt(comulativeV*comulativeV + comulativeH*comulativeH);
					if (comulative<0){
						comulative = 0;
					}else if (comulative>65535){
						comulative = 65535;
					}
				  //Limite the Combinacion to 8 bit size using Threshold in the scale of 16 bits
					if(comulative >= varThreshold16){
						comulative =255;
					}else{
						comulative = 0;
					}
					//Save the Combinacion of 8 bit to edge picture
					this->imageEdge.at<char>(r,c)	= ( char) (comulative); 
			}
		}
};


class Copy16 {
	private:
		cv::Mat &image;
    cv::Mat &imageTemp;

	public:
		Copy16(cv::Mat &img,cv::Mat &imgTemp) :image(img),imageTemp(imgTemp){}

		Copy16(Copy16 &x, tbb::split): image(x.image),imageTemp(x.imageTemp) {}

	  void operator() (const blocked_range<int> &r)const {
      //copi erosion16 image to original image
      for (int i = r.begin(); i != r.end(); i++) {
				int c = i / image.rows;
				int r = i % image.rows;
        image.at<unsigned short>(r, c) = imageTemp.at<unsigned short>(r, c);
        //ROS_INFO("R %i, C %i",r,c);
			}

		}

};

class Erosion8 {
	private:
		cv::Mat &image;
    cv::Mat &imageTemp;
    int maskR;

	public:
		Erosion8(cv::Mat &img,cv::Mat &imgTemp, int mask) :image(img), imageTemp(imgTemp),maskR(mask){}

		Erosion8(Erosion8 &x, tbb::split): image(x.image),imageTemp(x.imageTemp),maskR(x.maskR) {}
	void operator() (const blocked_range<int> &r)const {
    /***********************************
     * Erosion8:
     * Erosion for  edge image
     * 8 bits 1 canal
     * Add black pixels when pixels are near
     * *********************************/
    //Temp image to save erosrion changes
      bool dark;
			for (int i = r.begin(); i != r.end(); i++) {
				int c = i / image.rows;
				int r = i % image.rows;
        dark = false;
        //Get mask
        
        for (int tempi=r-maskR; tempi<(r+maskR);tempi++){
          if((tempi>0)&&(tempi<image.rows)){
            for (int tempj=c-maskR; tempj<(c+maskR);tempj++){
              //If found a drak pixel enable flag
              if((tempj>0)&&(tempj<image.cols)){
                if( this->image.at<char>(tempi, tempj) == 0){
                dark=true;
                }
              }else{
                dark=true;
              }
            }
          }else{
            dark=true;
          }
        }
        //If drak pixel eflag is nable then place dark pixel otherwise copy original image pixel
        if( dark){
            imageTemp.at<char>(r, c) = ( char)0;
          }else{
            imageTemp.at<char>(r, c) = ( char)255 ;
        }
        
			}

      
		}

};

class Dilation8 {
	private:
		cv::Mat &image;
    cv::Mat &imageTemp;
    int maskR;

	public:
		Dilation8(cv::Mat &img,cv::Mat &imgTemp, int mask) :image(img), imageTemp(imgTemp),maskR(mask){}

		Dilation8(Dilation8 &x, tbb::split): image(x.image),imageTemp(x.imageTemp),maskR(x.maskR) {}
	void operator() (const blocked_range<int> &r)const {


        /***********************************
     * Dilation8:
     * Dilation for  edge image
     * 8 bits 1 canal
     * Eliminate black pixels when white ixels are near
     * *********************************/

      bool white;
			for (int i = r.begin(); i != r.end(); i++) {
				int c = i / image.rows;
				int r = i % image.rows;
          white = false;
          //Get mask
          for (int tempi=r-maskR; tempi<(r+maskR);tempi++){
            if((tempi>0)&&(tempi<image.rows)){
              for (int tempj=c-maskR; tempj<(c+maskR);tempj++){
                if((tempj>0)&&(tempj<image.cols)){
                  //If found a drak pixel enable flag
                  if( this->image.at<char>(tempi, tempj) != 0){
                    white=true;
                  }
                }
              }
            }
          }
          //If drak pixel eflag is nable then place dark pixel otherwise whithe pixel
          if( white){
            imageTemp.at<char>(r, c) = ( char)255;
          }else{
            imageTemp.at<char>(r, c) = ( char)0 ;
          }
			}
		}
};

class Copy8 {
	private:
		cv::Mat &image;
    cv::Mat &imageTemp;

	public:
		Copy8(cv::Mat &img,cv::Mat &imgTemp) :image(img),imageTemp(imgTemp){}

		Copy8(Copy8 &x, tbb::split): image(x.image),imageTemp(x.imageTemp) {}

	  void operator() (const blocked_range<int> &r)const {
      //copi erosion16 image to original image
      for (int i = r.begin(); i != r.end(); i++) {
				int c = i / image.rows;
				int r = i % image.rows;
        this->image.at<char>(r, c) = imageTemp.at<char>(r, c);
        //ROS_INFO("R %i, C %i",r,c);
			}

		}

};








class ImageConverter
{
  //ROS
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  //image_transport::Publisher image_pub_;
  //Images 
    //Original
  cv::Mat image;
    //COnverted
  cv::Mat imageEdge;

  const int maskR=1;
  const int maskSize=1+(2*maskR);
  //Ros for dynamic reconfigure
  dynamic_reconfigure::Server<object_detection::object_detectionConfig> server;
  dynamic_reconfigure::Server<object_detection::object_detectionConfig>::CallbackType f;


public:
    int vardilation16=0;
    int varerosion16=2;
    int varopening16=0;
    int varclosing16=0;
    int varThreshold16=1000;
    //8 bits
    int vardilation8=0;
    int varerosion8=1;
    int varopening8=0;
    int varclosing8=0;
        long double msdilation16=0;
    long double mserosion16=2;
    long double msopening16=0;
    long double msclosing16=0;
    long double msThreshold16=1000;
    long double ms16=1000;
    //8 bits
    long double msdilation8=0;
    long double mserosion8=1;
    long double msopening8=0;
    long double msclosing8=0;
    long double msCounter=0;
    long double ms8=1000;
  ImageConverter(): it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/depth/image_rect_raw", 1, &ImageConverter::imageCb, this);
    //Publisher laser scan
    //image_pub_ = it_.advertise("/camera/depth/object_detection", 1);

    f = boost::bind(&ImageConverter::callback,this, _1, _2);
    server.setCallback(f);
    //Window to open image 
    cv::namedWindow(OPENCV_WINDOW);

  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }



  void erosion16(){
    /***********************************
     * Erosion16:
     * Erosion for  depth image
     * 16 bits 1 canal
     * Add black pixels when pixels are near
     * *********************************/
    //Temp image to save erosrion changes
    cv::Mat imageTemp (image.rows,image.cols,CV_16UC1);
    parallel_for(blocked_range<int>(0, (image.rows*image.cols)), Erosion16 (image,imageTemp,maskR));
    parallel_for(blocked_range<int>(0, (image.rows*image.cols)), Copy16(image,imageTemp));


  }
  void dilation16(){
    /***********************************
     * Dilation16:
     * Dilation for  depth image
     * 16 bits 1 canal
     * Eliminate black pixels when pixels are near
     * *********************************/
    cv::Mat imageTemp (image.rows,image.cols,CV_16UC1);
    parallel_for(blocked_range<int>(0, (image.rows*image.cols)), Dilation16 (image,imageTemp,maskR));
    parallel_for(blocked_range<int>(0, (image.rows*image.cols)), Copy16(image,imageTemp));
  }

  void opening16(){
      /***********************************
     * Opening16:
     * Opening for  depth image
     * 16 bits 1 canal
     * Get the opening of the depth images
     * *********************************/
    erosion16();
    dilation16();
  }
  void closing16(){
      /***********************************
     * Closing16:
     * closing for  depth image
     * 16 bits 1 canal
     * Get the closing of the depth images
     * *********************************/
    dilation16();
    erosion16();
  }
  void edge16() {
      this->imageEdge = cv::Mat(image.rows,image.cols,CV_8UC1);
      parallel_for(blocked_range<int>(0, (image.rows*image.cols)), Edge16 (image,imageEdge,varThreshold16));


		}

  void erosion8(){
    /***********************************
     * Erosion8:
     * Erosion for  edge image
     * 8 bits 1 canal
     * Add black pixels when pixels are near
     * *********************************/
    //Temp image to save erosrion changes
    cv::Mat imageTemp (this->imageEdge.rows,this->imageEdge.cols,CV_8UC1);
    parallel_for(blocked_range<int>(0, (imageEdge.rows*imageEdge.cols)), Erosion8(imageEdge,imageTemp,maskR));
    parallel_for(blocked_range<int>(0, (imageEdge.rows*imageEdge.cols)), Copy8(imageEdge,imageTemp));
  }



  void dilation8(){
    /***********************************
     * Dilation8:
     * Dilation for  edge image
     * 8 bits 1 canal
     * Eliminate black pixels when white ixels are near
     * *********************************/
    //Temp image to save dilation changes
    cv::Mat imageTemp (this->imageEdge.rows,this->imageEdge.cols,CV_8UC1);
    parallel_for(blocked_range<int>(0, (imageEdge.rows*imageEdge.cols)), Dilation8(imageEdge,imageTemp,maskR));
    parallel_for(blocked_range<int>(0, (imageEdge.rows*imageEdge.cols)), Copy8(imageEdge,imageTemp));
  }

  void opening8(){
      /***********************************
     * Opening8:
     * Opening for  depth image
     * 8 bits 1 canal
     * Get the opening of the depth images
     * *********************************/
    erosion8();
    dilation8();
  }
  void closing8(){
      /***********************************
     * Closing8:
     * closing for  depth image
     * 8 bits 1 canal
     * Get the closing of the depth images
     * *********************************/
    dilation8();
    erosion8();
  }





  void imageProcesingForDepthImage(){
      /***********************************
     * imageProcesingForDepthImage:
     * imageProcesingForDepthImage
     * 16 bits 1 canal to 8 bit canal
     * Cleans depth image and makes edge detection
     * *********************************/
    //Execute dilation16 n times

    start_timer();
    for(int i = 0;i<vardilation16;i++){
      dilation16();
    }
    msdilation16=msdilation16+stop_timer();
    start_timer();
    //Execute erosion16 n times
    for(int i = 0;i<varerosion16;i++){
      erosion16();
    }
    mserosion16=mserosion16+stop_timer();
    start_timer();
    //Execute opening16 n times
    for(int i = 0;i<varopening16;i++){
      opening16();
    }
    msopening16=msopening16+stop_timer();
    start_timer();
    //Execute closing16 n times
    for(int i = 0;i<varclosing16;i++){
      closing16();
    }
    msclosing16=msclosing16+stop_timer();
    start_timer();
    //Execute edge detection by a Threshold
    edge16();
    msThreshold16=msThreshold16+stop_timer();
    ms16= msdilation16+mserosion16+msopening16+msclosing16+msThreshold16;
  }



void imageProcesingForEdgeImage(){
      /***********************************
     * imageProcesingForEdgeImage:
     * imageProcesingForEdgeImage
     * 8 bits 1 canal
     * Cleans edge image and detecs obstacles
     * *********************************/
    //Execute dilation8 n times
    start_timer();
    for(int i = 0;i<vardilation8;i++){
      dilation8();
    }
    msdilation8=msdilation8+stop_timer();
    start_timer();
    //Execute opening8 n times
    for(int i = 0;i<varopening8;i++){
      opening8();
    }  
    msopening8=msopening8+stop_timer();
    start_timer();
    //Execute closing8 n times
    for(int i = 0;i<varclosing8;i++){
      closing8();
    }
    msclosing8=msclosing8+stop_timer();
    start_timer();
        //Execute erosion8 n times
    for(int i = 0;i<varerosion8;i++){
      erosion8();
    }   
    mserosion8=mserosion8+stop_timer();
    ms8 = msdilation8+msopening8+msclosing8+mserosion8;
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    /***********************************
     * imageCb:
     * imageCb is the code that is executed every time a new depth image is recived 
     * *********************************/

    // Try image convertion
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg),"16UC1";
      image     = cv_ptr->image;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    //Get the param fron the launcher
    //16 bits

    msCounter++;
    ROS_INFO("Total16= %Lf:Dilation16=%Lf,Erosion16=%Lf,Opening16==%Lf,Closing16=%Lf,Threshold16=%Lf",(ms16/msCounter),(msdilation16/msCounter),(mserosion16/msCounter),(msopening16/msCounter),(msclosing16/msCounter),(msThreshold16/msCounter));
    ROS_INFO("Total8= %Lf:Dilation8=%Lf,Erosion8=%Lf,Opening8==%Lf,Closing8=%Lf",(ms8/msCounter),(msdilation8/msCounter),(mserosion8/msCounter),(msopening8/msCounter),(msclosing8/msCounter));
    ROS_INFO("Total=%Lf",((ms8+ms16)/msCounter));
    //Make image procesing for depth image
    //16 bits
    imageProcesingForDepthImage();
    //o bits
    imageProcesingForEdgeImage();
    
    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, this->imageEdge);
    cv::waitKey(3);

    // Publisg modified video stream
    //    sensor_msgs::ImagePtr msgs = cv_bridge::CvImage(cv_ptr->header, "CV_8UC1", this->imageEdge).toImageMsg();
    //image_pub_.publish(msgs);
  }
  void callback(object_detection::object_detectionConfig &config, uint32_t level) {
  /*
  ROS_INFO("Reconfigure Request: %d %f %s %s %d", 
            config.int_param, config.double_param, 
            config.str_param.c_str(), 
            config.bool_param?"True":"False", 
            config.size);
            */
    vardilation16   =config.dilation16;
    varerosion16    =config.erosion16;
    varopening16    =config.opening16;
    varclosing16    =config.closing16;
    varThreshold16  =config.threshold16;
    //8 bits
    vardilation8  =config.dilation8;
    varerosion8   =config.erosion8;
    varopening8   =config.opening8;
    varclosing8   =config.closing8;
}

};

int main(int argc, char** argv)
{
  // Define ROS node
  ros::init(argc, argv, "image_converter");
  // Create a variable of ImageConverter class
  ImageConverter ic;








  // Execute the code in bucle
  ros::spin();
  return 0;
}