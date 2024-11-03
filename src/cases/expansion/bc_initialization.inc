    /* 
    * EXPERIMENTAL BOUNDARY CONDITIONS FOR SUDDEN EXPASION
    * X is periodic
    * there is a translation in z in order for the frontier to be periodic 
    * ===================
    * *******====********
    * *******====********
    * *******************
    * *******====********
    * *******====********
    * ===================
    * WARNING> NOT YET TESTED, 
    * 
    */
    
    int channel_half_Length =  5;
    int channel_half_height = 5;

    int z_translated = (z + (NZ_TOTAL-1)/2)%NZ_TOTAL;  


    // solid nodes around channel
    // north wall
    // south wall
    // east

    if (y == 0){ //S
        nodeType = 0b00110011;
    }else if (y == (NY - 1)){ // N
        nodeType = 0b11001100;
    }else{
        nodeType = 0b00000000;
    }

    if(z_translated <= channel_half_Length || z_translated >= (NZ_TOTAL-1)- channel_half_Length){
        nodeType == 0b11111111;         //solid
        //set lateral faces of the channel
        if (z_translated == channel_half_Length){ 
            nodeType = 0b00001111;            //wall 
            if (y == NY-1){
                nodeType == 0b11001111; //edge - concave
            }else if( y==0){
                nodeType == 0b00111111; //edge - concave
            }
        }else if (z_translated == (NZ_TOTAL-1)- channel_half_Length){
            nodeType = 0b11110000;           //wall 
            if (y == NY-1){
                nodeType == 0b11111100; //edge - concave
            }else if( y==0){
                nodeType == 0b11110011; //edge - concave
            }
        }
        if (y == (NY-1)/2 + channel_half_height){
            nodeType = 0b11001100;           //wall 
            if (z_translated == channel_half_Length){ 
                nodeType = 0b00001100;  //edge - convex
            }else if (z_translated == (NZ_TOTAL-1)- channel_half_Length){
                nodeType = 0b11000000; //edge - convex
            }
        }else if (y == (NY-1)/2 - channel_half_height){
            nodeType = 0b00110011;           //wall 
            if (z_translated == channel_half_Length){ 
                nodeType = 0b00000011;  //edge - convex
            }else if (z_translated == (NZ_TOTAL-1)- channel_half_Length){
                nodeType = 0b00110000; //edge - convex
            }
        }
        if((y > (NY-1)/2 - channel_half_height) && (y < (NY-1)/2 + channel_half_height)){
            nodeType = BULK;            //fluid
        }


    }
