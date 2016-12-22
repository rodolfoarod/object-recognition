CXX = g++
CXXFLAGS = -O2 -g -fmessage-length=0
CPPFLAGS = -I/usr/local/include

OBJS =      ./src/main.o ./src/svm.o

LDFLAGS = -L/usr/local/lib
LDLIBS =  -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video \
          -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect \
          -lopencv_contrib -lopencv_legacy -lopencv_gpu -lopencv_flann \
					-lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts \
					-lopencv_videostab -lopencv_nonfree

TARGET =    obj_rec_svm

.PHONY: all
all: $(TARGET)
$(TARGET):  $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
