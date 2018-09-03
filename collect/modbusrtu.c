/****************************************************
* This file control the gripper through modbus rtu
* Modified by xfyu on Jan 22
****************************************************/
#include "modbusrtu.h"
#include "modbustcp.h" // delay
#include "main.h"

/****************************************************
* Instruction Set
****************************************************/
unsigned char activate[] = {
	0x09, 0x10,
	0x03, 0xe8,
	0x00, 0x03,
	0x06,
	0x00, 0x00,
	0x00, 0x00,
	0x00, 0x00,
	0x73, 0x30
};

unsigned char read_gripper_status[] = {
	0x09, 0x03,
	0x07, 0xd0,
	0x00, 0x01,
	0x85, 0xcf
};

unsigned char activate_success[] = {
	0x09, 0x03,
	0x02,
	0x00, 0x00,
	0x59, 0x85
};

unsigned char close_with_certain_speed_certain_force[] = {
	0x09, 0x10,
	0x03, 0xe8,
	0x00, 0x03,
	0x06,
	0x09, 0x00,
	0x00, 0xff,
	0xff, 0xff,
	0x42, 0x29
};

unsigned char read_until_grip_completed[] = {
	0x09, 0x03,
	0x07, 0xd0,
	0x00, 0x03,
	0x04, 0x0e
};

unsigned char grip_is_completed1[] = {
	0x09, 0x03,
	0x02, 0xb9,
	0x00, 0x2a, 0x15
};

unsigned char grip_is_completed2[] = {
	0x09, 0x03,
	0x02, 0xf9,
	0x00, 0x1b, 0xd5
};

unsigned char open_with_certain_speed_certain_force[] = {
	0x09, 0x10,
	0x03, 0xe8,
	0x00, 0x03,
	0x06,
	0x09, 0x00,
	0x00, 0x00,
	0xff, 0xff,
	0x72, 0x19
};

unsigned char read_until_open_completed[] = {
	0x09, 0x03,
	0x07, 0xd0,
	0x00, 0x03,
	0x04, 0x0e
};

unsigned char open_is_completed[] = {
	0x09, 0x03,
	0x06,
	0xf9, 0x00,
	0x00, 0x00,
	0x03, 0x00,
	0x52, 0x2c
};

/****************************************************
* Functions
****************************************************/
/*
 * bufcmp - Compare the recv buf with what we 
 * already have
 *
 * Input: s1 - addr of the first buf
 *        s2 - addr of the second buf
 * Return Value: 0 - same
 *               1 - different
 * We don't have to know which buf is smaller, we
 * only care whether they are same.
 */
int bufcmp(unsigned char *s1, unsigned char *s2) {
	int len1 = sizeof(s1) - 1;
	int len2 = sizeof(s2) - 1;
	// printf("%d %d\n", len1, len2);
	if (len1 != len2)
		return 1; /* match fail */
	for (int i = 0; i < len1; ++i)
		if (s1[i] != s2[i])
			return 1;
	return 0;
}

/*
 * open_modbus - open the serial port
 *
 * Return Value: >0 - the fd. success
 *            	 <=0 - fail
 */
int open_modbus() {
	int fd;

	fd = open(MODBUS_DEV, O_RDWR);
	if (fd < 0) {
		perror("open tty error");
		return -1;
	}

	struct termios options;
	tcgetattr(fd, &options);
	memset(&options, 0, sizeof(options));
	// options.c_cflag |= CLOCAL | CREAD;
	options.c_cflag &= ~CSIZE;
	options.c_cflag |= CS8; /* 8 data bit */

	options.c_cflag &= ~PARENB; /* no parity */
	options.c_cflag &= ~CSTOPB; /* 1 stop bit */

	/* set the baudrate */
	if (cfsetispeed(&options, BAUDRATE) < 0) {
		perror("baudrate seti error");
		return -1;
	}
	if (cfsetospeed(&options, BAUDRATE) < 0) {
		perror("baudrate seto error");
		return -1;
	}
	/* set the wait time */
	options.c_cc[VTIME] = 10;
	options.c_cc[VMIN] = 4;

	/* bind the options to fd */
	if (tcsetattr(fd, TCSANOW, &options) < 0) {
		perror("attr set error");
		return -1;
	}

	return fd;
}
/*
 * gripper_activate - activate the gripper
 *
 * Return Value: 0 - success
 *               -1 - fail
 */
int gripper_activate() {
	int fd;
	int read_cnt;
	unsigned char recv_buf[BUF_SIZE];

	if ((fd = open_modbus()) < 0)
		return -1;

	/* activate */
	if (write(fd, activate, sizeof(activate)) < 0) {
		perror("write error");
		return -1;
	}

	int wait_cnt = 0;
	while (wait_cnt < 100000) {
		if (write(fd, read_gripper_status, 
				sizeof(read_gripper_status)) < 0) {
			perror("write error");
			return -1;
		}
		/* recv gripper status */
		if ((read_cnt = read(fd, recv_buf, BUF_SIZE)) < 0) {
			perror("read error");
			return -1;
		}

#ifdef DEBUG
		fprintf(stdout, "Activate Receive: ");
		for (int i = 0; i < read_cnt; ++i)
			fprintf(stdout, "0x%x ", recv_buf[i]);
		fprintf(stdout, "\n");
#endif
		wait_cnt++;
		if (!bufcmp(activate_success, recv_buf))
			break; /* complete */
		else
			continue; /* not complete */
	}
	if (wait_cnt >= 100000) {
		perror("wait activate error");
		return -1;
	}

	close(fd);
	return 0;
}

/*
 * gripper_close - close the gripper
 *
 * Input: speed: 0-255
 *		  force: 0-255
 * Return Value: 0 - success
 *               -1 - fail
 */
int gripper_close(unsigned char speed, unsigned char force) {
	int fd;
	int read_cnt;
	unsigned char recv_buf[BUF_SIZE];

	/* generate certain instruction */
	Generate_Open_Close_Instruct(speed, force);
	
	if ((fd = open_modbus()) < 0)
		return -1;

	/* grip */
	if (write(fd, close_with_certain_speed_certain_force, 
			sizeof(close_with_certain_speed_certain_force)) < 0) {
		perror("write error");
		return -1;
	}
	close(fd); /* close this connection */

	// printf("finish write cmd\n");
	delay(1000);
	int wait_cnt = 0;
	while (wait_cnt < 1000) {
		if ((fd = open_modbus()) < 0)
			return -1;
		// printf("fd is %d\n", fd);
		delay(1000);
		if (write(fd, read_gripper_status, 
				sizeof(read_gripper_status)) < 0) {
			perror("write error");
			return -1;
		}
		// printf("write status cmd complete! wait_cnt %d", wait_cnt);
		delay(1000);
		/* recv gripper status */
		if ((read_cnt = read(fd, recv_buf, BUF_SIZE)) < 0) {
			perror("read error");
			return -1;
		}
		close(fd);
#ifdef DEBUG
		fprintf(stdout, "Close Receive: ");
		for (int i = 0; i < read_cnt; ++i)
			fprintf(stdout, "0x%x ", recv_buf[i]);
		fprintf(stdout, "\n");
#endif
		wait_cnt++;
		// delay(1000); /* wait a little */
		if (!bufcmp(grip_is_completed1, recv_buf))
			return 0; /* complete 1 */
		else if (!bufcmp(grip_is_completed2, recv_buf))
			return 0; /* complete 2 */
		else
			continue; /* not complete */
	}
	perror("wait close error");
	return -1;
}

/*
 * gripper_open - open the gripper
 *
 * Input: speed: 0-255
 *		  force: 0-255
 * Return Value: 0 - success
 *               -1 - fail
 */
int gripper_open(unsigned char speed, unsigned char force) {
	int fd;
	int read_cnt;
	unsigned char recv_buf[BUF_SIZE];

	/* generate certain instruction */
	Generate_Open_Close_Instruct(speed, force);

	if ((fd = open_modbus()) < 0)
		return -1;

	/* open */
	if (write(fd, open_with_certain_speed_certain_force, 
			sizeof(open_with_certain_speed_certain_force)) < 0) {
		perror("write error");
		return -1;
	}

	int wait_cnt = 0;
	while (wait_cnt < 1000) {
		if (write(fd, read_until_open_completed, 
				sizeof(read_until_open_completed)) < 0) {
			perror("write error");
			return -1;
		}
		/* recv gripper status */
		if ((read_cnt = read(fd, recv_buf, BUF_SIZE)) < 0) {
			perror("read error");
			return -1;
		}
#ifdef DEBUG
		fprintf(stdout, "Open Receive: ");
		for (int i = 0; i < read_cnt; ++i)
			fprintf(stdout, "0x%x ", recv_buf[i]);
		fprintf(stdout, "\n");
#endif
		wait_cnt++;
		delay(1000); /* wait a little */
		if (!bufcmp(open_is_completed, recv_buf))
			break; /* complete */
		else
			continue; /* not complete */
	}
	if (wait_cnt >= 1000) {
		perror("wait open error");
		return -1;
	}

	close(fd);
	return 0;
}

/* 
 * ModBusCRC - compute the crc of a certain string
 *
 * Parameter: ptr - the start of that string
 *			  size - the length of that string
 * Return value: the crc result, a short. 
 */
unsigned short ModBusCRC(unsigned char * ptr, unsigned char size) {
	unsigned short a, b, tmp, CRC16;
	CRC16 = 0xffff; /* initiate CRC16 register value */

	for (a = 0; a < size; ++a) {
		CRC16 = *ptr ^ CRC16;
		for (b = 0; b < 8; ++b) {
			tmp = CRC16 & 0x0001;
			CRC16 >>= 1;
			if (tmp) /* check the bit that move out */
				CRC16 = CRC16 ^ 0xa001;
		}
		ptr++; /* move to the next byte */
	}
	return ((CRC16 & 0xFF) << 8) | ((CRC16 & 0xFF00) >> 8); 
}

/*
 * Generate_Open_Close_Instruct - generate the specific intruction
 * according to the Macro Definitions in main.h
 */
void Generate_Open_Close_Instruct(unsigned char speed, unsigned char force) {
	unsigned char length;
	unsigned short CRC;

	/* generate open instruction */
	open_with_certain_speed_certain_force[11] = speed;
	open_with_certain_speed_certain_force[12] = force;
	length = sizeof(open_with_certain_speed_certain_force) - 2;
	/* Calculate the CRC */
	CRC = ModBusCRC(open_with_certain_speed_certain_force, length);
	/* Add the CRC to the result */
	open_with_certain_speed_certain_force[13] = CRC >> 8;
	open_with_certain_speed_certain_force[14] = CRC & 0xff;

	/* generate close instruction */
	close_with_certain_speed_certain_force[11] = speed;
	close_with_certain_speed_certain_force[12] = force;
	/* Calculate the CRC, length is same */
	CRC = ModBusCRC(close_with_certain_speed_certain_force, length);
	/* Add the CRC to the result */
	close_with_certain_speed_certain_force[13] = CRC >> 8;
	close_with_certain_speed_certain_force[14] = CRC & 0xff;

#ifdef DEBUG
	length += 2;
	char buf[BUF_SIZE];
	sprintf(buf, "Open Instruction: ");
	for (int i = 0; i < length; ++i)
		sprintf(buf, "%s0x%x ", buf, open_with_certain_speed_certain_force[i]);
	fprintf(stdout, "%s\r\n", buf);

	sprintf(buf, "Close Instruction: ");
	for (int i = 0; i < length; ++i)
		sprintf(buf, "%s0x%x ", buf, close_with_certain_speed_certain_force[i]);
	fprintf(stdout, "%s\r\n", buf);
#endif
	return;
}

