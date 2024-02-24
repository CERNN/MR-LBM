#ifndef __NODE_TYPE_MAP_H
#define __NODE_TYPE_MAP_H

#include <builtin_types.h>
#include <stdint.h>


// DIRECTION DEFINES

#define BULK  (0b0000'0000)
//FACE
#define NORTH (0b1100'1100) //y=NY 
#define SOUTH (0b0011'0011) //y=0
#define WEST  (0b0101'0101)  //x=0
#define EAST  (0b1010'1010)  //x=NX
#define FRONT (0b1111'0000) //z=NZ
#define BACK  (0b0000'1111)  //z=0
//EDGE
#define NORTH_WEST  (0b1101'1101)
#define NORTH_EAST  (0b1110'1110)
#define NORTH_FRONT (0b1111'1100)
#define NORTH_BACK  (0b1100'1111)
#define SOUTH_WEST  (0b0111'0111)
#define SOUTH_EAST  (0b1011'1011)
#define SOUTH_FRONT (0b1111'0011)
#define SOUTH_BACK  (0b0011'1111)
#define WEST_FRONT  (0b1111'0101)
#define WEST_BACK   (0b0101'1111)
#define EAST_FRONT  (0b1111'1010)
#define EAST_BACK   (0b1010'1111)
//CORNER
#define NORTH_WEST_FRONT (0b1111'1101)
#define NORTH_WEST_BACK  (0b1101'1111)
#define NORTH_EAST_FRONT (0b1111'1110)
#define NORTH_EAST_BACK  (0b1110'1111)
#define SOUTH_WEST_FRONT (0b1111'0111)
#define SOUTH_WEST_BACK  (0b0111'1111)
#define SOUTH_EAST_FRONT (0b1111'1011)
#define SOUTH_EAST_BACK  (0b1011'1111)

#define SOLID_NODE (0b1111'1111)


#define BC_ZERO_VELOCITY_WALL (0b00000000'00000000'00000000'00000000) 
#define BC_VELOCITY_WALL      (0b00000000'00000000'00000001'00000000)
#define BC_OUTFLOW            (0b00000000'00000000'00000010'00000000)
#define BC_SYMMETRY           (0b00000000'00000000'00000011'00000000)
#define BC_FREESLIP           (0b00000000'00000000'00000100'00000000)
#define BC_EMPTY_ONE          (0b00000000'00000000'00000101'00000000)
#define BC_EMPTY_TWO          (0b00000000'00000000'00000111'00000000)

#define BC_VELOCITY_INDEX_0   (0b00000000'00000000'00000000'00000000)
#define BC_VELOCITY_INDEX_1   (0b00000000'00000000'00001000'00000000)
#define BC_VELOCITY_INDEX_2   (0b00000000'00000000'00010000'00000000)
#define BC_VELOCITY_INDEX_3   (0b00000000'00000000'00011000'00000000)


//    FEDCBAzy'xwvutsrq'ponmlkji'hgfedcba
// (0b00000000'00000000'00000000'00000000) 
// hgfedcba - nearby solid nodes (0 = fluid node, 255 = solid node)
// kji - boundary condition type
//      000 - solid wall
//      001 - fixed velocity (use index of ml for the velocities)
//      010 - outflow boundary condition (use index of ml for the pressure )
//      011 - symetry bondary condition
//      100 - free-slip boundary condition
//      101 - EMPTY
//      111 - EMPTY
// ml - velocity index (00,01,10,11)


#define DIRECTION_BITS (0b11111 << DIRECTION_OFFSET)





















#endif // !__NODE_TYPE_MAP_H
