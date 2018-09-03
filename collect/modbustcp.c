/**************************************************
* This file is for modbus tcp
* Modified by xfyu on April 13
**************************************************/
#include "modbustcp.h"
#include "modbusrtu.h" /* Check Wrist Bound need to open gripper */
#include "main.h"

/**************************************************
 * Global Variables
 **************************************************/
/* request frame for the tcp pose */
unsigned char pos_req_frm[12] = { 
	0x00, 0x01, /* sequence number */
	0x00, 0x00, /* protocol identifier */
	0x00, 0x06, /* package length */
	0x00, 0x04, /* function code for read input registers */
	0x01, 0x90, /* addr of first reg: 400 */
	0x00, 0x06  /* total number of reg to read */
};

/* request frame for the wrist angle */
unsigned char wrist_req_frm[12] = {
	0x00, 0x01, /* sequence number */
	0x00, 0x00, /* protocol identifier */
	0x00, 0x06, /* package length */
	0x00, 0x04, /* function code for read input registers */
	0x01, 0x0e, /* addr of first reg: 400 */
	0x00, 0x06  /* total number of reg to read */
};
/* wrist offset, to make the sign right */
const float OFFSET[] = {0, -6.283, 0, 0, 0, 0};

/***************************************************
* Functions
***************************************************/
/*
 * connect_modbus - establish the connection with modbus
 * Only open the connection, need the caller to close
 * the connection once finished.
 *
 * Parameter: none
 * Return value: >0 - success, the modbus_fd
 *               -1 - error
 */
int connect_modbustcp() {
	/* variables used in TCP connection */
	int clientSocket;
	struct sockaddr_in serverAddr;

	/* set up the socket of client */
	if ((clientSocket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		perror("socket error");
		return -1;
	}

	/* set the parameters */
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(MODBUS_PORT);
	serverAddr.sin_addr.s_addr = inet_addr(ROBOT_ADDR);
	/* try to connect to the server */
	if (connect(clientSocket, (struct sockaddr *)&serverAddr, 
			sizeof(serverAddr)) < 0) {
		perror("connect error");
		return -1;
	}
/*#ifdef DEBUG
	fprintf(stdout, "connect to port %d suceeded, clientSocket = %d\n", 
		MODBUS_PORT, clientSocket);
#endif*/

	return clientSocket;
}

/*
 * connect_realtime - establish the connection with robot server
 * Only open the connection, need the caller to close
 * the connection once finished.
 * Set the conn_sock.
 *
 * Parameter: none
 * Return value: >0 - success, the modbus_fd
 *               -1 - error
 */
int connect_realtime() {
	/* variables used in TCP connection */
	struct sockaddr_in serverAddr;
	int conn_sock;

	/* set up the socket of client */
	if ((conn_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		perror("socket error");
		return -1;
	}

	/* set the parameters */
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(REALTIME_PORT); /* the only diff */
	serverAddr.sin_addr.s_addr = inet_addr(REALTIME_ADDR);

#ifdef DEBUG
	printf("connecting...");
#endif

	/* try to connect to the server */
	if (connect(conn_sock, (struct sockaddr *)&serverAddr, 
			sizeof(serverAddr)) < 0) {
		perror("connect error");
		return -1;
	}
/* #ifdef DEBUG
	fprintf(stdout, "connect to port %d suceeded, conn_sock = %d\n", 
		REALTIME_PORT, conn_sock);
#endif*/

	return conn_sock;
}

/*
 * read_pos - Read 6 pos value
 * p[x, y, z, rx, ry, rz]. The x, y, z are in 0.1mm base.
 * The rx, ry, rz are in 0.001rad base. 
 *
 * Parameter: recv_value - the space to store the processed data
 * Return value: 0 - succeed
 *               nonzero - trasaction error
 *               1 - invalid data
 */
int read_pos(float * ret_value) {
	int clientSocket;
	/* receive buffer */
	unsigned char recvbuf[BUF_SIZE];
	short recv_value[6]; /* signed directly receive value */
	int res; /* ret num */
	short divider[2] = {10000, 1000};

	/* connect to the modbus server */
	if ((clientSocket = connect_modbustcp()) < 0)
		return -1;

	/* Send request */
	if (write(clientSocket, pos_req_frm, 
			sizeof(pos_req_frm)) < 0) {
		perror("send error");
		close(clientSocket);
		return 1;
	}

	/* store the size that receive */
	if ((res = read(clientSocket, recvbuf, BUF_SIZE)) < 0) {
		perror("receive error");
		close(clientSocket);
		return 1;
	}
/*#ifdef DEBUG
	fprintf(stdout, "Receive Bytes:");
	for (int i = 0; i < res; ++i) {
		fprintf(stdout, "%x ", recvbuf[i]);
	}
	fprintf(stdout, "\n");
#endif*/

	if (res >= 21) /* check if all 6 regs are read */
		for (int i = 0; i < REG_NUM; ++i) {
			int index = 9 + 2 * i; /* offset in the recvbuf */
			recv_value[i] = recvbuf[index] * 256 + 
				recvbuf[index + 1];
			/* process the value to its right base */
			ret_value[i] = recv_value[i] * 1.0 / divider[i>=3];
		}
	else {
		close(clientSocket);
		return 1;
	}

	close(clientSocket);
	return 0;
}

/*
 * read_wrist - Read 6 joint value
 * [base, shoulder, elbow, wrist1, wrist2, wrist3].
 * All values are in 0.001rad base. 
 *
 * Parameter: recv_value - the space to store the data
 * Return value: 0 - succeed
 *               nonzero - error
 *               1 - invalid data
 */
int read_wrist(float * ret_value) {
	int clientSocket;
	/* receive buffer */
	unsigned char recvbuf[BUF_SIZE];
	short recv_value[6]; /* signed directly receive value */
	int res; /* ret num */

	/* connect to the modbus server */
	if ((clientSocket = connect_modbustcp()) < 0)
		return -1;

	/* write request */
	if ((write(clientSocket, wrist_req_frm, 
			sizeof(wrist_req_frm))) < 0) {
		perror("send error");
		close(clientSocket);
		return 1;
	}

	/* store the size that receive */
	if ((res = read(clientSocket, recvbuf, BUF_SIZE)) < 0) {
		perror("receive error");
		close(clientSocket);
		return 1;
	}
#ifdef DEBUG
	fprintf(stdout, "Receive Bytes:");
	for (int i = 0; i < res; ++i) {
		fprintf(stdout, "%x ", recvbuf[i]);
	}
	fprintf(stdout, "\n");
#endif

	if (res >= 21) /* check if all 6 regs are read */
		for (int i = 0; i < REG_NUM; ++i) {
			int index = 9 + 2 * i; /* offset in the recvbuf */
			recv_value[i] = recvbuf[index] * 256 + recvbuf[index + 1];
			/* process the value to its base and add the offset */
			ret_value[i] = recv_value[i] * 1.0 / 1000 + OFFSET[i];
		}
	else {
		close(clientSocket);
		perror("invalid data");
		return 1; /* invalid data */
	}

	/* close the transaction */
	close(clientSocket);
	return 0;
}

/*
 * send_movel_instruct - send move intruction to robot
 *						Using position.
 *
 * Input: desired_pose - the target position
 * Return: 0 - success, -1 - failure
 *         Error message will display in the Msg box.
 */
int send_movel_instruct(float * desired_pose) {
	/* establish the realtime connection */
	int conn_sock = connect_realtime();
	if (conn_sock < 0)
		return -1; /* connect failed */
	char sendbuf[BUF_SIZE]; /* buffer to send */

	/* send the instrction */
	sprintf(sendbuf, "movel(p[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f])\r\n", 
		desired_pose[0], desired_pose[1], desired_pose[2], 
		desired_pose[3], desired_pose[4], desired_pose[5]);
	sprintf(sendbuf, "%s\r\n", sendbuf);
	if (write(conn_sock, sendbuf, sizeof(sendbuf)) < 0) {
		perror("Send instruction error");
		return -1;
	}
	close(conn_sock);

#ifdef DEBUG
	fprintf(stdout, "instruction:%s", sendbuf);
#endif

	/* wait until the move complete */
	float * cur_pose = (float *)malloc(REG_NUM * sizeof(float));
	read_pos(cur_pose);
	int cnt = 0;
	while ((wait_until_nodiff(cur_pose, desired_pose) > THRESHOLD) &&
		cnt <= 10000) {
		read_pos(cur_pose);
		cnt++;
		// delay(10000);
	}
	if (cnt > 10000)
		perror("Move failed!");
	
	free(cur_pose);
	return 0;
}

/*
 * send_movej_instruct - send move intruction to robot
 *						Using joint positions.
 * move to the position decribed by angles' values
 *
 * Input: float * desired_joint - target position
 * Return: 0 - success, -1 - failure
 *         Error message will display in the Msg box.
 */
int send_movej_instruct(float * desired_joint) {
	char sendbuf[BUF_SIZE]; /* buffer to send */

	/* establish the realtime connection */
	int conn_sock = connect_realtime();
	if (conn_sock < 0)
		return -1; /* connect failed */
	/* send the instrction */
	sprintf(sendbuf, "movej([%.3f,%.3f,%.3f,%.3f,%.3f,%.3f])\r\n", 
		desired_joint[0], desired_joint[1], desired_joint[2], 
		desired_joint[3], desired_joint[4], desired_joint[5]);
	sprintf(sendbuf, "%s\r\n", sendbuf);
	if (write(conn_sock, sendbuf, sizeof(sendbuf)) < 0) {
		perror("Send instruction error");
		return -1;
	}
	close(conn_sock);
#ifdef DEBUG
	fprintf(stdout, "instruction:%s", sendbuf);
#endif
	/* wait until the move complete */
	float *cur_joint = (float *)malloc(REG_NUM*sizeof(float));
	int cnt = 0;
	read_wrist(cur_joint);
	while ((wait_until_nodiff(cur_joint, desired_joint) > THRESHOLD) &&
		cnt <= 10000) {
		int res = read_wrist(cur_joint);
		delay(1000);
		cnt++;
	}
	if (cnt > 10000)
		perror("Move failed!");

	// printf("movej complete!\n");
	free(cur_joint);
	return 0;
}

/*
 * send_movej_screw - send move intruction to robot
 *						Using joint positions.
 * Adjust the 6th wrist.
 *
 * Input: type - UP or DOWN. 
 * Return: none.
 *         Error message will display in the Msg box.
 */
void send_movej_screw(int direction) {
	char sendbuf[BUF_SIZE]; /* buffer to send */
	delay(1000); /* wait for a little */
	Check_Wrist_Bound(direction, FOCUS_STEP); /* check first */
	delay(1000); /* wait for a little */
	float * cur_joint = (float *)malloc(REG_NUM * sizeof(float));
	read_wrist(cur_joint); /* read the newest value */
	/* apply for a temporary place */
	float * desired_joint = (float *)malloc(REG_NUM * sizeof(float));
	/* copy the current joint value */
	memcpy(desired_joint, cur_joint, REG_NUM * sizeof(float));
	/* set the desired_joint according to type */
	desired_joint[5] += direction * FOCUS_STEP; 

	send_movej_instruct(desired_joint);

	// printf("screw complete!\n");
	free(desired_joint);
	free(cur_joint);
	return;
}

/*
 * move_from_to - move last wrist for delta_angle in direction
 *
 * Input: type - UP or DOWN. 
 *        delta_angle - the abs of moving angle
 * Return: none.
 *         Error message will display in the Msg box.
 */
void move_from_to(int direction, float delta_angle) {
	/* check for the range */
	if (delta_angle < 0) {
		perror("delta angle less than 0!");
		return;
	}
	char sendbuf[BUF_SIZE]; /* buffer to send */

	float * cur_joint = (float *)malloc(REG_NUM * sizeof(float));
	/* apply for a temporary place */
	float * desired_joint = (float *)malloc(REG_NUM * sizeof(float));

	/* compute the next move */
	float next_move;
	while (delta_angle > 0) {
		/* the smaller one */
		next_move = delta_angle > MAX_SCREW_ANGLE ? MAX_SCREW_ANGLE : delta_angle;
		delta_angle -= next_move;
		// fprintf(stdout, "%.3f, %.3f\n", next_move, delta_angle);
		delay(1000);
		Check_Wrist_Bound(direction, next_move); /* check first */
		delay(1000);
		read_wrist(cur_joint); /* read the newest value */
		/* copy the current joint value */
		memcpy(desired_joint, cur_joint, REG_NUM * sizeof(float));
		/* set the desired_joint according to type */
		desired_joint[5] += direction * next_move;
		/* send the instruct */
		if(send_movej_instruct(desired_joint) < 0) /* fails */
			break;
	}
	// printf("move from to complete!\n");
	free(desired_joint);
	free(cur_joint);
	return;
}

/*
 * Check_Wrist_Bound - check whether wrist 3 has reached the limitation
 * If next joint reaches max, turn to min and then start the next move
 * If next joint reaches min, turn to max and then start the next move
 * Only check bound and move to the MAX or MIN if necessary.
 * Not responsible for the execution of next move.
 * 
 * Input: direction - UP or DOWN
 * 		  next_move - a positive value, the abs of changed angle
 */
void Check_Wrist_Bound(int direction, float next_move) {
	char sendbuf[BUF_SIZE]; /* buffer to send */
	
	/* read current joint */
	float *cur_joint = (float *)malloc(REG_NUM * sizeof(float));
	read_wrist(cur_joint);

	/* apply for a temporary place, used in move instruction */
	float *desired_joint = (float *)malloc(REG_NUM * sizeof(float));
	/* copy the current joint value */
	memcpy(desired_joint, cur_joint, REG_NUM * sizeof(float));
	/* check whether reach the max or min */
	fprintf(stdout, "cur %.3f,next %.3f,sum %.3f\n", cur_joint[5], next_move, cur_joint[5] + direction*next_move);
	if (direction == UP && cur_joint[5] + next_move > MAX_ANGLE) {
		if (gripper_open(SPEED_DEFAULT, FORCE_DEFAULT) < 0) {
			free(desired_joint);
			free(cur_joint);
			return; /* return if fails */
		}
		/* set the desired_joint according to type */
		desired_joint[5] = MIN_ANGLE; /* move to min */

		if(send_movej_instruct(desired_joint) < 0) { /* fails */
			free(desired_joint);
			free(cur_joint);
			return;
		}
		fprintf(stdout, "REACH UPPER BOUND! desired joint:[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]\n", 
		desired_joint[0], desired_joint[1], desired_joint[2], 
		desired_joint[3], desired_joint[4], desired_joint[5]);

		if (gripper_close(SPEED_DEFAULT, FORCE_DEFAULT) < 0) {
			free(desired_joint);
			free(cur_joint);
			return;
		}
	}
	else if (direction == DOWN && cur_joint[5] - next_move < MIN_ANGLE) {
		if (gripper_open(SPEED_DEFAULT, FORCE_DEFAULT) < 0) {
			free(desired_joint);
			free(cur_joint);
			return; /* return if fails */
		}
		/* set the desired_joint according to type */
		desired_joint[5] = MAX_ANGLE; /* move to max */

		if(send_movej_instruct(desired_joint) < 0) { /* fails */
			free(desired_joint);
			free(cur_joint);
			return;
		}
		fprintf(stdout, "REACH LOWER BOUND! desired joint:[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]\n", 
		desired_joint[0], desired_joint[1], desired_joint[2], 
		desired_joint[3], desired_joint[4], desired_joint[5]);

		if (gripper_close(SPEED_DEFAULT, FORCE_DEFAULT) < 0) {
			free(desired_joint);
			free(cur_joint);
			return;
		}
	}
	// fprintf(stdout, "finish checkbound!\n");
	free(desired_joint);
	free(cur_joint);
	return;
}

/*
 * wait_until_nodiff - calculate the difference between 2 vectors
 */
float wait_until_nodiff(float *a1, float *a2) {
	float sum = 0;
	/* calculate the ^2 sum */
	for (int i = 0; i < REG_NUM; ++i) {
		sum += (a1[i] - a2[i]) * (a1[i] - a2[i]);
		// fprintf(stdout, "*a1 %f, *a2 %f\n", a1[i], a2[i]);
	}
#ifdef DEBUG
	fprintf(stdout, "sum is %f\n", sum);
#endif
	return sum;
}

/*
 * delay - delay some time
 */
void delay(int n) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < 1000; j++)
			;
}
