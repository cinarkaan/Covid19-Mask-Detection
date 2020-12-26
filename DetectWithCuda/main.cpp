#pragma once
#include <iostream>
#include <string.h>
#include <WS2tcpip.h> 
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

sockaddr_in _init_() {
	WSADATA data; // Bilgiler burada tutulacak
	WORD version = MAKEWORD(2, 2); // Versiyon numarasý ve winsock bilgileri veriliyor (standart)
	int wsok = WSAStartup(version, &data);
	if (wsok != 0) {
		cout << " --(!)Can not start winsock  " << endl;
	}

	sockaddr_in server; // Soketin baðlanacaðý yer elemanlar vs bilgileri tutulur
	server.sin_family = AF_INET; // Standarttýr
	server.sin_port = htons(5002); // Kullaýlacak port no
	inet_pton(AF_INET, "127.0.0.1", &server.sin_addr); // Kullanýlacak ip adresi
	return server;
}

char buffer[2]; // Serverdan gelen durum saklanýr
string _result[2]; // Maske durum bilgisi
string message_length, length_str;
sockaddr_in server = _init_(); // Soket yapýlandýrýlýyor
GpuMat frame_Gray_Cuda, faces, cudagray; // Cuda icin resim saklama sýnýfý
Mat face_roi; // Normal resim saklama sýnýfý
vector<Rect> _faces; // Belirlenen yüzü saklamak için vektor sýnýfý
vector<uchar> buf;

void getGpuDriverInfo() {
	DeviceInfo _deviceInfo;
	printShortCudaDeviceInfo(getDevice());
	int cuda_devices_number = getCudaEnabledDeviceCount();
	cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;

	bool _isd_evice_compatible = _deviceInfo.isCompatible();
	cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
}

void turnImg() {
	resize(face_roi, face_roi, Size(50, 50));
	buf.resize(face_roi.rows * face_roi.cols / 2);
	imencode(".jpg", face_roi, buf);
	length_str = to_string(buf.size());
	message_length = string(16 - length_str.length(), '0') + length_str;
}

int sendAndrecv(SOCKET out) {
	sendto(out, (const char*)message_length.c_str(), 16, 0, (sockaddr*)&server, sizeof(server)); // Gönderilen resim boyutu
	sendto(out, (const char*)buf.data(), buf.size(), 0, (sockaddr*)&server, sizeof(server)); // Gönderilen resim
	recv(out, buffer, sizeof(buffer), 0); // Maske durumu serverdan geliyor
	return buffer[0] - 48; // Ascii tablosuna göre int tipine çevir döndür
}

void maskDetection(Mat frame, Ptr<cuda::CascadeClassifier> face_Cascade_Gpu, SOCKET out) {
	frame_Gray_Cuda.upload(frame);
	cv::cuda::cvtColor(frame_Gray_Cuda, frame_Gray_Cuda, cv::COLOR_BGR2GRAY);
	face_Cascade_Gpu->detectMultiScale(frame_Gray_Cuda, faces); // Cascade dosyasýndaki hedeflenen kareyi(yüzü) yakala
	face_Cascade_Gpu->convert(faces, _faces); // GpuMat objesini 2d vectore dönüþtür

	for (size_t i = 0; i < _faces.size(); i++) // Dikdörgen çiz hedeflenen karenin üstüne yerleþtir
	{
		frame_Gray_Cuda(_faces[i]).download(face_roi);
		turnImg();
		int result = sendAndrecv(out); // Servere bilgiyi gönder ve cavabý al
		putText(frame, _result[result], Point(_faces[i].x, _faces[i].y - 10), FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 255, 0, 0), 1, 8, false); // Maske durumunu bas
		rectangle(frame, _faces[i], Scalar(0, 0, 255)); // Yuzu dikdörgen içine al
	}
	cv::imshow("Capture - Face detection", frame);
}

int main() {
	const string video_path = "C:\\Users\\Kaan\\Documents\\MaskData\\testvideo\\facemasktestvideo.mp4";
	const String face_cascade_name = "C:\\Users\\Kaan\\Documents\\HaarCascadeMask\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml"; 
	
	Mat frame;
	VideoCapture capture;
	Ptr<cuda::CascadeClassifier> face_Cascade_Gpu = cuda::CascadeClassifier::create(face_cascade_name); 

	SOCKET out = socket(AF_INET, SOCK_DGRAM, 0); 

	capture.open(video_path);
	face_Cascade_Gpu->setScaleFactor(1.25f); // Tespit edilen yüzün hangi oranda yeniden ölçeklenmesini saðlayan method
	face_Cascade_Gpu->setMinNeighbors(4); // Cascade yapýsýndaki pozitiflerin ve negatifleri biraz oynatan metod
	face_Cascade_Gpu->setMinObjectSize(Size(48, 48)); // Tespit edilen yüz 48*48 boyutundan küçükse dikkate alma
	_result[0] = "Mask";
	_result[1] = "NoMask";
	getGpuDriverInfo();

	while (capture.read(frame)) {
		maskDetection(frame, face_Cascade_Gpu, out);
		if (waitKey(27) == 27)
			break;
	}

	closesocket(out);

	WSACleanup();

	return 1;
}
