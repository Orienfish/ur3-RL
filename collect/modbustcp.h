/**************************************************
* This file is for modbus tcp
* Modified by xfyu on Jan 22
**************************************************/
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "main.h"
#include "modbusrtu.h"

#define ROBOT_ADDR "192.168.0.1" /* server addr */
#define MODBUS_PORT 502 /* server port */
#define REG_NUM 6 /* num of reg to read */

/* for realtime socket connection */
#define REALTIME_ADDR "192.168.0.1"
#define REALTIME_PORT 30003 /* real time client */

/**************************************************
* Function Definitions
**************************************************/
int connect_modbustcp();
int connect_realtime();
int read_pos(float * recv_value);
int read_wrist(float * recv_value);
int send_movel_instruct(float * desired_pose);
int send_movej_instruct(float * desired_joint);
void send_movej_screw(int direction);
void Check_Wrist_Bound(int direction, float next_move);
void move_from_to(int direction, float delta_angle);
float wait_until_nodiff(float *a1, float *a2);
void delay(int n);
