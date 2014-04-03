/**************************************************
** TimerManager.hpp				                       **
** ---------------------                         **
**                                               **
** Manage two king of timer : glutTimer and      **
**	HighResolutionTimer (from Herubel Adrien)    **
**                                               **
** Note: This is almost certainly broken under   **
**    Cygwin...                                  **
**                                               **
**************************************************/


/* Related defined data type:                                            */
/* --------------------------                                            */
/*                                                                       */
/* timerStruct                                                           */
/*        (Under windows this is a nasty, nasty union, under Linux it's  */
/*         a struct timespec, under MacOS, it's currently a struct       */
/*         timespec, but I'm not sure if that'll work).                  */

/* Defined functions:                                                    */
/* ------------------                                                    */
/*                                                                       */
/* inline void GetHighResolutionTime( timerStruct *t );                  */
/*        Gets the current time and store it in the timerStruct.  This   */
/*        should have nanosecond resolution on all systems (except maybe */
/*        Cygwin), though the OS may not update the timer ever ns.       */
/*                                                                       */
/* inline float ConvertTimeDifferenceToSec( timerStruct *end,            */
/*                                          timerStruct *begin );        */
/*        Takes two timerStructs and returns a floating point value      */
/*        representing the seconds between the begin and end point.      */
/*        Beware using this over long periods of time (> 0.1 sec) if you */
/*        *really* want nanosecond precision, as your float will lose    */
/*        precision in those bits -- reimplement with a double or an int */



#ifndef __TIMER_MANAGER_HPP
#define __TIMER_MANAGER_HPP


extern "C" {

/* This header defines two functions and one data type used by them,     */
/*    but the implmentation changes, depending on which system is being  */
/*    used.  You may need to modify the #ifdef's to work correctly  on   */
/*    your computer...  I defined them rather arbitrarily, based on what */
/*    has worked for me (or in the case of the MacOS, a guess).          */

// Are we using MS Visual Studio?
#if defined(WIN32) && defined(_MSC_VER)
	#define USING_MSVC
#endif

// Are we using MacOS?
#if defined(__APPLE__)
	#define USING_MACOSX
#endif

// Are we using GCC under Linux or other Unixes (including Cygwin)?
#if defined(__GNUC__) && !defined(USING_MACOSX)
	#define USING_LINUX
#endif

#include <GL/glut.h>

/* the following mess implements these timing mechanisms  */

#if defined(USING_MSVC)  // Use code that works on MS Visual Studio.

        /* this code should link without any work on your part */

	#include <windows.h>
	#pragma comment(lib, "kernel32.lib")
	typedef LARGE_INTEGER timerStruct;
	inline void GetHighResolutionTime( timerStruct *t )
		{ QueryPerformanceCounter( t ); }
	inline float ConvertTimeDifferenceToSec( timerStruct *end, timerStruct *begin )
		{ timerStruct freq; QueryPerformanceFrequency( &freq );  return (end->QuadPart - begin->QuadPart)/(float)freq.QuadPart; }
	inline float ConvertTimeDifferenceNowToSec( timerStruct *now, timerStruct *last )
		{ timerStruct freq; QueryPerformanceFrequency( &freq ); QueryPerformanceCounter( now ); return (now->QuadPart - last->QuadPart)/(float)freq.QuadPart;}

#elif defined(USING_LINUX)  // Assume we have POSIX calls clock_gettime()

        /* on some Linux systems, you may need to link in the realtime library (librt.a or librt.so) in
	   order to use this code.  You can do this by including -lrt on the gcc/g++ command line.
	*/

	#include <time.h>
	typedef struct timespec timerStruct;
	inline void GetHighResolutionTime( timerStruct *t )
		{ clock_gettime( CLOCK_REALTIME, t ); }
	inline float ConvertTimeDifferenceToSec( timerStruct *end, timerStruct *begin )
		{ return (end->tv_sec - begin->tv_sec) + (1e-9)*(end->tv_nsec - begin->tv_nsec); }
	inline float ConvertTimeDifferenceNowToSec( timerStruct *now, timerStruct *last )
		{ clock_gettime( CLOCK_REALTIME, now); return (now->tv_sec - last->tv_sec) + (1e-9)*(now->tv_nsec - last->tv_nsec);}
#elif defined(USING_MACOSX)  // Assume we're running on MacOS X

	/* this code uses calls from the CoreServices framework, so to get this to work you need to
	   add the "-framework CoreServices" parameter g++ in the linking stage. This code was adapted from:
	   http://developer.apple.com/qa/qa2004/qa1398.html
	*/

	#include <CoreServices/CoreServices.h>
	#include <mach/mach.h>
	#include <mach/mach_time.h>
	typedef uint64_t timerStruct;
	inline void GetHighResolutionTime( timerStruct *t )
		{ *t = mach_absolute_time(); }
	inline float ConvertTimeDifferenceToSec( timerStruct *end, timerStruct *begin )
		{ uint64_t elapsed = *end - *begin; Nanoseconds elapsedNano = AbsoluteToNanoseconds( *(AbsoluteTime*)&elapsed );
			return float(*(uint64_t*)&elapsedNano) * (1e-9); }
	inline float ConvertTimeDifferenceNowToSec( timerStruct *now, timerStruct *last )
		{ uint64_t elapsed = mach_absolute_time() - *begin; Nanoseconds elapsedNano = AbsoluteToNanoseconds( *(AbsoluteTime*)&elapsed ); return float(*(uint64_t*)&elapsedNano) * (1e-9); }
#endif


	typedef struct timer_properties {
		unsigned int	type;	// 0 means GLUT timer and >0 means HighResolutionTimer
		unsigned int	last_start_glut;		// Stockage temps précédent pour la version GLUT
		timerStruct		last_start_hres;		// Stockage temps précédent pour la version High Resolution Timer
		float					last_execution;			// Temps d'execution de la dernière execution
		bool					is_running;					// On enregistre le temps sur ce timer (entre un start et un stop)
	} timerProperties;
} //end extern "C"


class TimerManager {
	/// \TODO Faire des hashtable avec unsigned int ou des noms...
	public:
		TimerManager() {resetAllTimers();}

		void resetAllTimers() {nb_timers=0;elapsed_times_sec.clear();nb_executions.clear();timers.clear();}
		unsigned int addOneTimer(unsigned int newtypetimer = 0) {
			elapsed_times_sec.push_back(0.);
			nb_executions.push_back(0);
			timerProperties prop;
			prop.type = newtypetimer;
			prop.last_execution = 0.0;
			timers.push_back(prop);
			return nb_timers++;
		}

		/*
		void removeOneTimer(unsigned int the_removed_timer) {
			if(the_removed_timer>=nb_timers) {
				exit(1);
				// ERROR...
			}
			elapsed_times_sec.erase( elapsed_times_sec.begin()+the_removed_timer, elapsed_times_sec.begin()+the_removed_timer);
			nb_executions.erase( nb_executions.begin()+the_removed_timer, nb_executions.begin()+the_removed_timer);
			timers.erase( timers.begin()+the_removed_timer, timers.begin()+the_removed_timer);
			nb_timers--;
		}
		*/

		inline void startOneExecution(unsigned int id_timer,bool gl_finish = false) {
			nb_executions[id_timer]++;
			timers[id_timer].is_running = true;
			if (gl_finish) glFinish();
			if (timers[id_timer].type == 0) {
				timers[id_timer].last_start_glut = glutGet(GLUT_ELAPSED_TIME);
			}
			else {
				GetHighResolutionTime( &(timers[id_timer].last_start_hres) );
			}
		}

		inline void stopOneExecution(unsigned int id_timer,bool gl_finish = false) {
			if (gl_finish) glFinish();
			if (timers[id_timer].type == 0) {
				timers[id_timer].last_execution = (glutGet(GLUT_ELAPSED_TIME) - timers[id_timer].last_start_glut)*0.001;
			}
			else {
				timers[id_timer].last_execution = ConvertTimeDifferenceNowToSec(&now,&(timers[id_timer].last_start_hres));
			}
			elapsed_times_sec[id_timer] += timers[id_timer].last_execution;
			timers[id_timer].is_running = false;
		}

		void listTimers() {
			for( unsigned int i = 0; i<nb_timers;i++ ) {
				std::cout << "Timer "<<i<<" : "<<elapsed_times_sec[i]<<" for "<<nb_executions[i]<<" executions"<<std::endl;
			}
		}

		inline float getLastTime(unsigned int id_timer) {
			if(id_timer>=nb_timers) exit(1); // Generate ERROR
			return timers[id_timer].last_execution*1000.0;
		}

		inline float getAveragedTime(unsigned int id_timer) {
			if(id_timer>=nb_timers) exit(1); // Generate ERROR
			return elapsed_times_sec[id_timer]*1000.0/(float)nb_executions[id_timer];
		}

		inline unsigned int retrieveNbExecution(unsigned int id_timer) {
			if(id_timer>=nb_timers) exit(1); // Generate ERROR
			return nb_executions[id_timer];
		}

		inline void resetOneTimer(unsigned int id_timer) {
			if(id_timer>=nb_timers) exit(1); // Generate ERROR
			if (timers[id_timer].is_running) {
				std::cerr<<"Timer is running !"<<std::endl; // Or generate an ERROR
			}
			nb_executions[id_timer] = 0;
			elapsed_times_sec[id_timer] = 0.;
		}


		unsigned int nb_timers;

	private:
		timerStruct now;															///< Variable temporaire servant à éviter les allocations
		std::vector<float> elapsed_times_sec;					///< Liste des temps d'exécutions
		std::vector<unsigned int> nb_executions;			///< Nombre d'éxecutions
		std::vector<timerProperties> timers;					///< Les structures de sauvegarde du temps
};

#endif // end #ifndef HIGHRES_TIMER_HPP
