CC := dpcpp
CXX := dpcpp
RM := rm -rf

CFLAGS := -g -Wall -Wno-deprecated-declarations -fPIC -std=c++17 -fsycl

OUTPUT = list-compute-devices

OBJS := list-compute-devices.o

all: $(OUTPUT)

clean:
	$(RM) $(OBJS) $(OUTPUT)	

$(OUTPUT): $(OBJS)
	$(CXX) $(CFLAGS) -o $(OUTPUT) $(OBJS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c -o $@ $<

.PHONY: all clean
