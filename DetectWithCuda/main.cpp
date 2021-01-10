#include "ServerStream.h"

int main() {
	int frameCount = 0;
	const string video_path = "C:\\Users\\Kaan\\Documents\\MaskData\\testvideo\\facemasktestvideo2.mp4";
	const String face_cascade_name = "C:\\Users\\Kaan\\Documents\\HaarCascadeMask\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml";

	Mat frame;
	VideoCapture capture;
	ServerStream serverstream(5002, "127.0.0.1",face_cascade_name);
	
	serverstream.setCascade(1.435f, 2, Size(48, 48));
	capture.open(video_path);
	 
	while (capture.read(frame)) {
		serverstream.detectfacemask(frame);
		cout << "Frame : " << frameCount++ << " Mask : " << serverstream.getMaskCount() << " NoMask : " << serverstream.getNoMaskCount() << endl;
		serverstream.reset();
		if (waitKey(27) == 27)
			break;
				
	}

	return 1;

}