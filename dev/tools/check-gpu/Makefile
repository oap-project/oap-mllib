CC := clang
CXX := clang++
RM := rm -rf

CFLAGS := -g -Wall -Wno-deprecated-declarations -fPIC -std=c++17 -fsycl

OUTPUT = check-gpu

OBJS := check-gpu.o

all: $(OUTPUT)

clean:
	$(RM) $(OBJS) $(OUTPUT)	

$(OUTPUT): $(OBJS)
	$(CXX) $(CFLAGS) -o $(OUTPUT) $(OBJS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c -o $@ $<

.PHONY: all clean
