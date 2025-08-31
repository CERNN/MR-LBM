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
#define MACR_SAVE (10) // save macroscopic fields every # timestgeps
#define REPORT_SAVE (false) // save report data every # timestgeps
#define VTK_SAVE (true) // save in vtk file
#define BIN_SAVE (false) //save in binary file

//#define BC_FORCES //create scalar field to export the reaction forces from BC;


/* ------------------------- TIME AND SAVE DEFINES ------------------------- */

#define PARTICLES_SAVE (10)               // Save particles info every given steps (0 not report)
#define PARTICLES_DATA_REPORT (false)                   // Report IBM treated data every given steps (0 not report)


#define IBM_PARTICLES_NODES_SAVE true      // Saves particles nodes data
/* ------------------------------------------------------------------------- */