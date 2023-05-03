#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 


enum flag {continued, gameover, win, quit} FLAG; //flag用于控制游戏进程

pthread_mutex_t move_mutex;
int Thread_id[9] = {0,1,2,3,4,5,6,7,8}; //9个thread控制log



struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN] ; 

void printMap(void){
	puts("\033[H\033[2J");
	for(int i = 0; i <= ROW; ++i)	
		puts( map[i] );
}

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


flag check_over(int x, int y){ //用于更新flag控制游戏进程
	if (x==0){
		return win; //win
	}
	else if (y >= COLUMN -1 || y<=0){
		return gameover; //lose
	}
	else if (map[x][y]==' '){
		return gameover; //lose
	}
	else if (map[x][y]=='='){
		return continued;
	}
	else{
		return continued; 
	}
}
//在按下Q时单独设置quit

void *logs_move( void *t ){
	/*  Move the logs  */
	int  Tid = *(int*)t; //储存每一个线程的TID，这样才知道是哪一个线程
	(Tid)++; //传进来的值可能为0，无法判断奇数偶数，所以加1	


	while(FLAG == continued){
		usleep(100000); //每次程序挂起的时间，来控制log移动的速度
		pthread_mutex_lock(&move_mutex);
		int moved = 0;
		int eachlen = 0;
		for (int i = 0; i <COLUMN-1; i++){//先遍历计算每个log的长度
			if (map[Tid][i]=='=' || map[Tid][i]=='0'){
				eachlen++;
			}	
		}
		// printf("The log length of %dth log is %d\n", Tid, eachlen);
		// printf("this is tid: %d\n", Tid);

		//偶数的TID向右移动
		if (Tid % 2==0){
			for (int i = COLUMN - 2; i > -1; i--){
				//移动log
				if (map[Tid][i]==' ' &&  map[Tid][(i-1+COLUMN-1)%(COLUMN-1)]=='='){//从右向左判断到第一个为空且前一个为log的位置，则为头（唯一存在的情况，例外处理如下）
					if (!moved){//若到了边界会出现判断两次的问题，所以添加flag让每次遍历只动一次
					map[Tid][i]='=';
					map[Tid][(i-eachlen+COLUMN-1)%(COLUMN-1)] = ' ';
					moved = 1;
					}
			
				}
				if (map[Tid][i]=='0'){
					if (map[Tid][(i+1)%(COLUMN-1)]==' '){ //如果frog在头,即frog的右边为空（向右移动）
						map[Tid][(i+1)%(COLUMN-1)]=='0';
						map[Tid][i]=='=';
						map[Tid][(i-eachlen+COLUMN-1)%(COLUMN-1)] = ' ';
					}
					else{
						map[Tid][(i+1)%(COLUMN-1)]='0'; //frog在中间或者尾部
						map[Tid][i]='=';
					}
					frog.y++; //更新frog的位置
					if (FLAG != quit){ //若当前状态不为quit 则更新flag
						FLAG = check_over(frog.x, frog.y);//如果已经是quit，则不改变flag，使线程不再更改flag而继续
					}; 
				}

			}
		}
		//奇数的TID向左移动
		else{
			for (int j = 0; j <COLUMN-1; j++){//向左移动则从左向右开始判断
				if ((map[Tid][j]==' ') &&  (map[Tid][(j+1)%(COLUMN-1)]=='=')){ //遇到第一个空格左边为log即为头（唯一存在的情况，例外处理如下）
					if (!moved){ //若到了边界会出现判断两次的问题，所以添加flag让每次遍历只动一次
					map[Tid][(j)%(COLUMN-1)]='=';
					map[Tid][(j+eachlen)%(COLUMN-1)] = ' ';
					moved = 1;
					} //只改变头尾的log和空格
				}

				if (map[Tid][j]=='0'){//如果frog在头,即frog的左边为空（向右移动）
					if (map[Tid][(j+48)%(COLUMN-1)]==' '){
						map[Tid][(j+48)%(COLUMN-1)]=='0';
						map[Tid][j]=='=';
						map[Tid][(j+eachlen-1+COLUMN-1)%(COLUMN-1)] = ' ';
					}
					else{
						map[Tid][(j+48)%(COLUMN-1)]='0'; //frog在中间或者尾部
						map[Tid][j]='=';
					}
					frog.y--;//更新frog的位置
					if (FLAG != quit){//同上
						FLAG = check_over(frog.x, frog.y);
					}; 
				}
			}
		}

		printMap();
		if (FLAG != quit){
			FLAG = check_over(frog.x, frog.y);
		}//再次更新位置
		pthread_mutex_unlock(&move_mutex);

	}
	pthread_exit(NULL);	//退出thread
}


void *frog_move( void *t ){
	while (FLAG == continued){
		pthread_mutex_lock(&move_mutex);

		if (kbhit()){
			char movement = getchar();
			/*1st----向前动*/
			if (movement == 'W' || movement == 'w'){
				// int checkflag;
				// if (FLAG == continued) {checkflag =1;};
				if (check_over(frog.x-1,frog.y) == continued){//判断下一步向前如果能走且状态为continue
					if(frog.x == ROW){							//在起始点则走了之后填充‘|’
						map[frog.x][frog.y] = '|';
					}
					else{										//在log则走了之后填充‘=’
						map[frog.x][frog.y] = '=';
					}
					map[--frog.x][frog.y] = '0' ;               //更新frog的位置并且表示在map上
				}
				else if(check_over(frog.x-1,frog.y) == win){	//判断下一步是否win
					map[frog.x][frog.y] = '=';
					map[--frog.x][frog.y] = '0' ;
					FLAG = check_over(frog.x,frog.y);
					pthread_mutex_unlock(&move_mutex);
				}
				else{
					// frog.x-=1;
					if (frog.x != ROW){
						map[frog.x][frog.y] = '=';
					}
					else{
						map[frog.x][frog.y] = '|';
					}
					map[--frog.x][frog.y] = '0';								
					FLAG = check_over(frog.x,frog.y);			//若下一步不是continue或者win，则已经不能继续，更新状态
				}
				pthread_mutex_unlock(&move_mutex);
			}
			/*2nd----向后动*/
			else if (movement == 'S' || movement == 's'){  //思路同上
				if (frog.x != ROW){

				if (check_over(frog.x+1,frog.y)==continued){
					if (frog.x == 0){
						map[frog.x][frog.y] = '|' ;
					}
					else{
						map[frog.x][frog.y] = '=' ;
					}
					map[++frog.x][frog.y] = '0' ;
				}
				else{
					// frog.x++;
					map[frog.x][frog.y] = '=';
					map[++frog.x][frog.y] = '0';								
					FLAG = check_over(frog.x,frog.y);			
				}
				}
				pthread_mutex_unlock(&move_mutex);
			}
			/*3rd----向左动*/
			else if(movement == 'A' || movement == 'a'){      //思路同上， 不过不需要判断win的情况
				if (check_over(frog.x, frog.y-1) == continued){
					if (frog.x == ROW){
							map[frog.x][frog.y] = '|';
					}
					else{
						// if (map[frog.x][frog.y])
						map[frog.x][frog.y] = '=';
					}
					map[frog.x][--frog.y] = '0';
				}
				else{
					map[frog.x][--frog.y] = '0';
					FLAG = check_over(frog.x,frog.y-1);
					
				
				}
				pthread_mutex_unlock(&move_mutex);
			}
			/*4th----向右动*/
			else if(movement == 'D' || movement == 'd'){    //思路同上， 不过不需要判断win的情况
				if (check_over(frog.x, frog.y+1) == continued){
					if (frog.x == ROW){
							map[frog.x][frog.y] = '|';
					}
					else{
						map[frog.x][frog.y] = '=';
					}
					map[frog.x][++frog.y] = '0';
				}
				else{
					map[frog.x][++frog.y] = '0';
					FLAG = check_over(frog.x,frog.y+1);	
				}
				pthread_mutex_unlock(&move_mutex);
			}
			/*5th----退出游戏quit*/
			else if (movement == 'Q' || movement == 'q'){
				FLAG = quit; //直接将flag设置为quit
				pthread_mutex_unlock(&move_mutex);
				pthread_exit(NULL);

			}
		}

		pthread_mutex_unlock(&move_mutex);
	}
	pthread_exit(NULL);
}

void print_state(flag f){  //根据flag情况来输出信息
	if (f == gameover){
		printf("You lose the game!!");
	}
	else if (f == win){
		printf("You win the game!!");
	}
	else if (f == quit){
		printf("You exit the game.");
	}
}


int main( int argc, char *argv[] ){
	FLAG = continued;
	// srand(time(NULL));
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	//初始化log的位置
	for (int i = 1; i<ROW; i++){
		int minlen = 10;
		int maxlen = 15;
		int randomlength = rand() % (COLUMN);
		int range = rand() % maxlen +minlen;
		//随机生成每一块木板的长度
		for (int j = randomlength; j < range+ randomlength; j++){
			map[i][j%(COLUMN-1)] = '=';	
		}

	}


	/*  Create pthreads for wood move and frog control.  */

	int x=0; //传入frog thread参数所需
	int rc; //依照tutorial思路创建thread
	pthread_t Thread_log[9], Thread_frog;
	pthread_mutex_init(&move_mutex,NULL);

	//create threads for each log
	for (int i = 0; i < 9; i++){
		rc = pthread_create(&Thread_log[i], NULL, logs_move, (void*)&Thread_id[i]);
		if (rc){
            printf("ERROR: return code from pthread_create() is %d", rc);
            exit(1);
	}
	}
	//create thread for frog
	rc = pthread_create(&Thread_frog, NULL, frog_move, (void*)&x);
	if (rc){
		printf("ERROR: return code from pthread_create() is %d", rc);
		exit(1);
	}
	//join thread
	for (int i=0; i< 9; i++){
        pthread_join(Thread_log[i],NULL);
    }

	pthread_join(Thread_frog, NULL);


	printMap();
	puts("\033[H\033[2J");
	print_state(FLAG);

	pthread_mutex_destroy(&move_mutex);
	pthread_exit(NULL);

	return 0;

}
