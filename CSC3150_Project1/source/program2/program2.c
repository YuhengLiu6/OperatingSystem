#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

// wait_opts structure
struct wait_opts {
  enum pid_type wo_type;
  int wo_flags;
  struct pid *wo_pid;
  struct waitid_info *wo_info;
  int wo_stat;
  struct rusage *wo_rusage;
  wait_queue_entry_t child_wait;
  int notask_error;
};
static struct task_struct *task;

extern pid_t kernel_clone(struct kernel_clone_args *kargs);

extern struct filename *getname_kernel(const char __user *filename);
extern struct filename *getname(const char __user *filename);
extern long do_wait(struct wait_opts *wo);
extern int do_execve(struct filename *filename,
                     const char __user *const __user *__argv,
                     const char __user *const __user *__envp);
// process termination signals
char *TerminationSignal[] = {"null",   "SIGHUP",  "SIGINT",  "SIGQUIT",
                             "SIGILL", "SIGTRAP", "SIGABRT", "SIGBUS",
                             "SIGFPE", "SIGKILL", NULL,      "SIGSEGV",
                             NULL,     "SIGPIPE", "SIGALRM", "SIGTERM"};

/* If WIFEXITED(STATUS), the low-order 8 bits of the status.  */
#define __WEXITSTATUS(status) (((status)&0xff00) >> 8)

/* If WIFSIGNALED(STATUS), the terminating signal.  */
#define __WTERMSIG(status) ((status)&0x7f)

/* If WIFSTOPPED(STATUS), the signal that stopped the child.  */
#define __WSTOPSIG(status) __WEXITSTATUS(status)

/* Nonzero if STATUS indicates normal termination.  */
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)

/* Nonzero if STATUS indicates termination by a signal.  */
#define __WIFSIGNALED(status) (((signed char)(((status)&0x7f) + 1) >> 1) > 0)

/* Nonzero if STATUS indicates the child is stopped.  */
#define __WIFSTOPPED(status) (((status)&0xff) == 0x7f)

pid_t pid = 0;

// implement wait function
void my_wait(pid_t pid) {
  printk("[program2] : child process\n");

  int status;
  struct wait_opts wo;
  struct pid *wo_pid = NULL;
  enum pid_type type;
  type = PIDTYPE_PID;
  wo_pid = find_get_pid(pid);

  wo.wo_type = type;
  wo.wo_pid = wo_pid;
  wo.wo_flags = WEXITED | WUNTRACED;
  wo.wo_info = NULL;
  wo.wo_stat = 10;
  wo.wo_rusage = NULL;

  int a;
  a = do_wait(&wo);
  put_pid(wo_pid);

  // check return status
  int returnsig = (unsigned long)wo.wo_stat;
  /*normal termination*/
  if (__WIFEXITED(returnsig)) {
    printk("[program2] : Child process gets NORMAL termination and return "
           "signal is %d\n",
           returnsig);
  }
  /*child process stopped*/
  else if (__WIFSTOPPED(returnsig)) {
    int STOP = __WSTOPSIG(returnsig);
    printk("[program2] : get SIGSTOP signal\n");
    printk("[program2] : child process terminated");
    printk("[program2] : The return signal is %d", STOP);
  } else if (__WIFSIGNALED(returnsig)) {
    int TERMINATE = __WTERMSIG(returnsig);
    if (TerminationSignal[TERMINATE] != NULL) {
      printk("[program2] : get %s signal\n", TerminationSignal[TERMINATE]);
    }
    printk("[program2] : child process terminated");
    printk("[program2] : The return signal is %d", TERMINATE);
  } else {
    printk("[program2] : Child process continues\n");
  }

  return;
}

int my_exec(void) {
  int rlt;
  char path[] = "/tmp/test";
//   const char path[] =
//   	"/home/vagrant/csc3150/Assignment_1_120090263/program1/normal";
  struct filename *myfile = getname_kernel(path);
  rlt = do_execve(myfile, NULL, NULL);
  if (!rlt) {
    return 0;
  } else {
    printk("[program2] : The result of my_exec() has errors!");
  }

  do_exit(rlt);
}

// implement fork function
int my_fork(void *argc) {
  // set default sigaction for current process
  int i;
  struct k_sigaction *k_action = &current->sighand->action[0];
  for (i = 0; i < _NSIG; i++) {
    k_action->sa.sa_handler = SIG_DFL;
    k_action->sa.sa_flags = 0;
    k_action->sa.sa_restorer = NULL;
    sigemptyset(&k_action->sa.sa_mask);
    k_action++;
  }

  /* fork a process using kernel_clone or kernel_thread */
  struct kernel_clone_args args = {
      .flags =
          ((lower_32_bits(SIGCHLD) | CLONE_VM | CLONE_UNTRACED) & ~CSIGNAL),
      .exit_signal = (lower_32_bits(SIGCHLD) & CSIGNAL),
      .stack = (unsigned long)my_exec,
      .stack_size = (unsigned long)NULL,
  };

  pid = kernel_clone(&args);
  printk("[program2] : The child process has pid = %d", pid);
  printk("[program2] : This is the parent process, pid =  %d", current->pid);
  /* execute a test program in child process */
  my_wait(pid);
  /* wait until child process terminates */
  return 0;
}

static int __init program2_init(void) {
  printk("[program2] : Module_init {Liu_Yuheng} {ID:120090263}\n");

  /* write your code here */

  /* create a kernel thread to run my_fork */
  printk("[program2] : Module_init create kthread start\n");
  task = kthread_create(&my_fork, NULL, "MyThread");

  // wake up new thread if ok
  if (!IS_ERR(task)) {
    printk("[program2] : Module_init kthread start\n");
    wake_up_process(task);
  }

  return 0;
}

static void __exit program2_exit(void) { printk("[program2] : Module_exit\n"); }

module_init(program2_init);
module_exit(program2_exit);
