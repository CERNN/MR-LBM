/* ----------------------------- OUTPUT DEFINES ---------------------------- */
#define ID_SIM "000"            // prefix for simulation's files
#define PATH_FILES "TEST"  // path to save simulation's files
#define TREATFIELD (false) //treat data over the entire field
#define TREATPOINT (false) //treat data in a single or several points
#define TREATLINE (false) //save the macro in a line
#define SAVEDATA (false) //save treat data
#define CONSOLEPRINT (false) // print the console the data is being saved
#define MEAN_FLOW (false) // store the mean flow of the domain (used to calculate turbulent statistics)
#define NODE_TYPE_SAVE (false) //save the bc conditions, usefull for drawing the surface
#define ONLY_FINAL_MACRO (false) //save only the last time step macroscopic
#define MACR_SAVE (false)
#define REPORT_SAVE (false)
#define VTK_SAVE (true)
#define BIN_SAVE (false)

//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin


/* ------------------------- TIME AND SAVE DEFINES ------------------------- */

#define IBM_PARTICLES_SAVE (100)               // Save particles info every given steps (0 not report)
#define IBM_DATA_REPORT (100)                   // Report IBM treated data every given steps (0 not report)

 
#define IBM_DATA_STOP false                 // stop condition by IBM treated data
#define IBM_DATA_SAVE false                 // save reported IBM data to file

#define IBM_PARTICLES_NODES_SAVE true      // Saves particles nodes data
/* ------------------------------------------------------------------------- */
