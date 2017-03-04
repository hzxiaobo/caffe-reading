CC      = gcc
CFLAGS  += -fPIC -lpthread
CSHARED += -fPIC -shared
OPENCV_DIR=/usr
OPENCV_INCLUDE=-I$(OPENCV_DIR)/include/opencv -I$(OPENCV_DIR)/include/opencv2 -I/usr/java/jdk1.8.0_45/include -I/usr/java/jdk1.8.0_45/include/linux
OPENCV_LIBS=-L$(OPENCVTWO_DIR)/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_features2d -lopencv_calib3d  -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -lopencv_video
BIN_DIR = .
CAFFE_INCLUDE = -I/home/xiaobo/work/caffe-master/include -I/usr/local/cuda-7.0/include -I/home/xiaobo/work/caffe-master/build/src
CAFFE_LIBS = -L/home/xiaobo/work/caffe-master/build/lib -lcaffe
GENERATE_DIR = ./
GENERATE_LIBS = -L$(GENERATE_DIR) 
OBJ	= caffeclassifier.o imgbaseprocess.o classjni.o fileprocess.o
SSO	= libimg_classify_target.so
BIN	= testcaffe

all: $(BIN) libimg_classify_target.so

libimg_classify_target.so: $(OBJ)
	$(CXX) $(CSHARED) -o $(GENERATE_DIR)/$@ $(OBJ) $(CAFFE_LIBS) $(OPENCV_LIBS)

testcaffe:libimg_classify_target.so test_caffe_classify.cpp caffeclassifier.h classjni.h fileprocess.h
	$(CXX) $(CFLAGS) $(CAFFE_INCLUDE) $(OPENCV_INCLUDE) test_caffe_classify.cpp -o $(BIN_DIR)/$@ $(CAFFE_LIBS) $(OPENCV_LIBS) $(GENERATE_LIBS) $(SSO)
	
caffeclassifier.o: caffeclassifier.cpp caffeclassifier.h
	$(CXX) $(CFLAGS) $(CAFFE_INCLUDE) $(OPENCV_INCLUDE) -c caffeclassifier.cpp -o $(BIN_DIR)/$@ $(CAFFE_LIBS) $(OPENCV_LIBS)

imgbaseprocess.o: imgBaseProcess.cpp imgBaseProcess.h
	$(CXX) $(CFLAGS) $(CAFFE_INCLUDE) $(OPENCV_INCLUDE) -c imgBaseProcess.cpp -o $(BIN_DIR)/$@ $(CAFFE_LIBS) $(OPENCV_LIBS)

classjni.o: classjni.cpp classjni.h
	$(CXX) $(CFLAGS) $(CAFFE_INCLUDE) $(OPENCV_INCLUDE) -c classjni.cpp -o $(BIN_DIR)/$@ $(CAFFE_LIBS) $(OPENCV_LIBS)

fileprocess.o: fileprocess.cpp fileprocess.h
	$(CXX) $(CFLAGS) $(CAFFE_INCLUDE) $(OPENCV_INCLUDE) -c fileprocess.cpp -o $(BIN_DIR)/$@ $(CAFFE_LIBS) $(OPENCV_LIBS)

clean:
	rm -f *.o
	rm -f *.cpp~

.PHONY: clean
