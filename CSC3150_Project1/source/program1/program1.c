#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
// Find corresponding signals from header files
char *TerminationSignal[] = { "null", "SIGHUP",  "SIGINT",  "SIGQUIT",
			      "SIGILL",   "SIGTRAP", "SIGABRT", "SIGBUS",
			      "SIGFPE",   "SIGKILL", NULL,      "SIGSEGV",
			      NULL,       "SIGPIPE", "SIGALRM", "SIGTERM" };

int main(int argc, char *argv[])
{
	/* fork a child process */
	pid_t pid, childpid;
	int status;

	printf("Process starts to fork\n");
	pid = fork();

	if (pid == 1) {
		perror("fork");
		exit(1);
	} else {
		// Child
		// process---------------------------------------------------------------------------------------------------
		if (pid == 0) {
			int i;
			char *arg[argc];

			printf("I'm the CHILD Process: pid = %d\n", getpid());
			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;

			/* execute test program */
			printf("Child process start to execute test program:\n");
			execve(arg[0], arg, NULL);
			// raise(SIGCHLD);
			exit(EXIT_FAILURE);
		} else {
			printf("I'm the FATHER Process: pid = %d\n", getpid());

			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);
			/* check child process'  termination status */
			printf("This is termination status:\n");

			// Terminate by normal exit
			if (WIFEXITED(status)) {
				printf("Parent process receives SIGCHLD signal\n");
				printf("Normal termination with Exit status: %d \n",
				       WEXITSTATUS(status));
			}

			// Child process is paused
			else if (WIFSTOPPED(status)) {
				if (WSTOPSIG(status) == SIGSTOP) {
					printf("Parent process receives SIGSTOP signal , status: %d\n",
					       WSTOPSIG(status));
					printf("CHILD PROCESS STOPPED\n");
				} else {
					printf("Child process STOPPED with signal out of sample: %d\n",
					       WSTOPSIG(status));
				}
			}

			// Terminate due to different signals
			else if (WIFSIGNALED(status)) {
				// check conditions to print cooresponding signals
				if (TerminationSignal[WTERMSIG(status)] !=
				    NULL) {
					printf("Parent process receives  %s signal\n",
					       TerminationSignal[WTERMSIG(
						       status)]);
				} else {
					printf("CHILD process FAILED with signal out of sample\n");
				}
				printf("CHILD Execution Failed: %d\n",
				       WTERMSIG(status));
			}

			// CHILD process continued
			else {
				printf("Child process CONTINUED.\n");
			}
		}
		// end of father process-------------------------
		exit(0);
	}
	return 0;
}
