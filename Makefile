CC=g++
CFLAGS=-std=c++11

CSV_HOME=$(HOME)/soft/csv
INCLUDES=-I $(CSV_HOME)/include
TARGET=read_channelized_data

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDES)

sanity_check: sanity_check.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDES)

plot:
	python3 -m pipenv run python plot_channelized_data.py

clean:
	rm $(TARGET) $(TARGET).o
