#pragma once
#include <iostream>
#include <string.h>
#include <WS2tcpip.h> 
#include <omp.h>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>

#pragma comment (lib, "ws2_32.lib") 

using namespace std;
using namespace cv;
using namespace cv::cuda;

class ServerStream
{

private:
	int result,mask,nomask;
	char buffer[2];
	string _result[2];
	string message_length, length_str;
	GpuMat faces,frame_Gray_Cuda;
	SOCKET out;
	sockaddr_in server;
	vector<uchar> buf;
	sockaddr_in _init_(int port, string servername);
	Ptr <cuda::CascadeClassifier> face_Cascade_Gpu;

	void turnImg(Mat face_roi);

	void getGpuDriverInfo();

	void sendAndrecv();

public:

	ServerStream(int port,string servername,string face_cascade_name) {
		getGpuDriverInfo();
		this->server = _init_(port, servername);
		this->out = socket(AF_INET, SOCK_DGRAM, 0);
		this->_result[0] = "Mask";
		this->_result[1] = "NoMask";
		this->face_Cascade_Gpu = cuda::CascadeClassifier::create(face_cascade_name);
	}

	~ServerStream() {
		closesocket(out);
		WSACleanup();
	}

	void detectfacemask(Mat frame);

	void setCascade(float scalefactor, int mineightbor, Size minobjectsize);

	void reset() {
		this->mask = 0;
		this->nomask = 0;
	}

	int getMaskCount() {
		return this->mask;
	}

	int getNoMaskCount() {
		return this->nomask;
	}
};

void ServerStream::getGpuDriverInfo() {
	DeviceInfo _deviceInfo;
	printShortCudaDeviceInfo(getDevice());
	int cuda_devices_number = getCudaEnabledDeviceCount();
	cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;

	bool _isd_evice_compatible = _deviceInfo.isCompatible();
	cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
}

void ServerStream::detectfacemask(Mat frame) {
	vector<Rect> _faces;
	frame_Gray_Cuda.upload(frame);
	cuda::cvtColor(frame_Gray_Cuda, frame_Gray_Cuda, cv::COLOR_BGR2GRAY);
	face_Cascade_Gpu->detectMultiScale(frame_Gray_Cuda, faces); // Cascade dosyasýndaki hedeflenen kareyi(yüzü) yakala
	face_Cascade_Gpu->convert(faces, _faces); // GpuMat objesini 2d vectore dönüþtür
	
	#pragma omp parallel for shared(_faces) schedule(dynamic,4)
	for (int i = 0; i < _faces.size(); i++) // Dikdörgen çiz hedeflenen karenin üstüne yerleþtir
	{
		turnImg(frame(_faces[i])); // Resmi dönüstür
		sendAndrecv(); // Servere bilgiyi gönder ve cavabý al
		result ? ++nomask : ++mask; 
		putText(frame, _result[result], Point(_faces[i].x, _faces[i].y - 10), FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 255, 0, 0), 1, 8, false); // Maske durumunu bas
		rectangle(frame, _faces[i], Scalar(0, 255, 0)); // Yuzu dikdörgen içine al
	}
	imshow("Capture - MaskDetection", frame);
}

void ServerStream::setCascade(float scalefactor, int mineightbor, Size minobjectsize) {
	ServerStream::face_Cascade_Gpu->setScaleFactor(scalefactor);
	ServerStream::face_Cascade_Gpu->setMinNeighbors(mineightbor);
	ServerStream::face_Cascade_Gpu->setMinObjectSize(minobjectsize);
	ServerStream::face_Cascade_Gpu->setFindLargestObject(false);
}
 
void ServerStream::turnImg(Mat detected) {
	resize(detected, detected, Size(50, 50));
	buf.resize(detected.rows * detected.cols / 2);
	imencode(".jpg", detected, buf);
}

void ServerStream::sendAndrecv() {
	//sendto(out, (const char*)message_length.c_str(), 16, 0, (sockaddr*)&server, sizeof(server)); // Gönderilen resim boyutu
	sendto(out, (const char*)buf.data(), buf.size(), 0, (sockaddr*)&server, sizeof(server)); // Gönderilen resim 
	recv(out, buffer, sizeof(buffer), 0); // Maske durumu serverdan geliyor
	result =  buffer[0] - 48; // Ascii tablosuna göre int tipine çevir döndür
}

sockaddr_in ServerStream::_init_(int port, string servername) {
	WSADATA data; // Bilgiler burada tutulacak
	WORD version = MAKEWORD(2, 2); // Versiyon numarasý ve winsock bilgileri veriliyor (standart)
	int wsok = WSAStartup(version, &data);
	if (wsok != 0) {
		cout << " --(!)Can not start winsock  " << endl;
	}
	sockaddr_in server; // Soketin baðlanacaðý yer elemanlar vs bilgileri tutulur
	server.sin_family = AF_INET; // Standarttýr
	server.sin_port = htons(port); // Kullaýlacak port no
	inet_pton(AF_INET, servername.c_str(), &server.sin_addr); // Kullanýlacak ip adresi
	return server;
}

