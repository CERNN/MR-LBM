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
#define MACR_SAVE (5000)
#define REPORT_SAVE (false)
#define VTK_SAVE (true)
#define BIN_SAVE (false)


//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin

