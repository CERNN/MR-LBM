#ifndef __NODE_TYPE_MAP_H
#define __NODE_TYPE_MAP_H

#include <builtin_types.h>
#include <stdint.h>


// DIRECTION DEFINES

#define BULK  (0b00000)
//FACE
#define NORTH (0b00001) //y=NY
#define SOUTH (0b00010) //y=0
#define WEST (0b00011)  //x=0
#define EAST (0b00100)  //x=NX
#define FRONT (0b00101) //z=NZ
#define BACK (0b00110)  //z=0
//EDGE
#define NORTH_WEST (0b00111)
#define NORTH_EAST (0b01000)
#define NORTH_FRONT (0b01001)
#define NORTH_BACK (0b01010)
#define SOUTH_WEST (0b01011)
#define SOUTH_EAST (0b01100)
#define SOUTH_FRONT (0b01101)
#define SOUTH_BACK (0b01110)
#define WEST_FRONT (0b01111)
#define WEST_BACK (0b10000)
#define EAST_FRONT (0b10001)
#define EAST_BACK (0b10010)
//CORNER
#define NORTH_WEST_FRONT (0b10011)
#define NORTH_WEST_BACK (0b10100)
#define NORTH_EAST_FRONT (0b10101)
#define NORTH_EAST_BACK (0b10110)
#define SOUTH_WEST_FRONT (0b10111)
#define SOUTH_WEST_BACK (0b11000)
#define SOUTH_EAST_FRONT (0b11001)
#define SOUTH_EAST_BACK (0b11010)

#define DIRECTION_BITS (0b11111 << DIRECTION_OFFSET)





















#endif // !__NODE_TYPE_MAP_H
