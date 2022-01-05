#ifndef __NODE_TYPE_MAP_H
#define __NODE_TYPE_MAP_H

#include <builtin_types.h>
#include <stdint.h>


// DIRECTION DEFINES
#define DIRECTION_BITS (0b11111 << DIRECTION_OFFSET)
#define BULK  (0b00000)
#define NORTH (0b00001) //y=NY
#define SOUTH (0b00010) //y=0
#define WEST (0b00011)  //x=0
#define EAST (0b00100)  //x=NX
#define FRONT (0b00101) //z=NZ
#define BACK (0b00110)  //z=0


#endif // !__NODE_TYPE_MAP_H
