/****************************************************
* This file control the gripper through modbus rtu
* Modified by xfyu on Jan 22
****************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include "main.h"

#define MODBUS_DEV "/dev/ttyUSB0"
#define BAUDRATE B115200

/* already defined otherwhere */
// #define BUF_SIZE 512

// #define DEBUG

/**************************************
* Function Definitions
**************************************/
int bufcmp(unsigned char *s1, unsigned char *s2);
int open_modbus();
int gripper_activate();
int gripper_close(unsigned char speed, unsigned char force);
int gripper_open(unsigned char speed, unsigned char force);
unsigned short ModBusCRC(unsigned char * ptr, unsigned char size);
void Generate_Open_Close_Instruct(unsigned char speed, unsigned char force);
