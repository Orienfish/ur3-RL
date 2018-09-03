/********************************************************************************
** main.h - necessary definitions
**
** Created by xfyu on Jan 19, 2018
********************************************************************************
/* necessary macro definitions */
// #define DEBUG

#define BUF_SIZE 512 /* maximum rev length */
#define REG_NUM 6 /* 6 values for 1 pose */
#define PICNAMESIZE 256

/* Used in read_pose */
#define FOR_CUR 0
#define FOR_COARSE 1
#define FOR_FINE 2

/* choices of LastOP */
#define UP 1
#define DOWN -1

/* choices of coarse_or_fine */
#define COARSE 0
#define FINE 1
#define FOCUS_POS_FILE "init_pos.txt"

/* The focus step */
#define FOCUS_STEP 0.3

/* Control the gripper */
/* Speed and force are set from python side */
#define FORCE_DEFAULT 20
#define SPEED_DEFAULT 40

/* Threshold used in judging whether last
   operation is finished */
#define THRESHOLD 0.005

/* min and max limitation of wrist 3 */
#define MIN_ANGLE 0.1
#define MAX_ANGLE 6.2
#define MAX_SCREW_ANGLE 6.0 /* familiar with FOCUS_STEP */

/* one loop coarse = TIMES * one loop fine */
#define TIMES 9


