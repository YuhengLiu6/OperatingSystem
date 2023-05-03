#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 gstorage_ptr = 0;
__device__ __managed__ u32 ADDRESS_BASE = 128;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  /* Overview of FCB------
    0 : valid bit
    1-20: file name
    21-23: file size
    24-25: create time
    26-27: modified time
    28-31: file address  /存的是在storage中的block number
  */
  //init valid bit to 0
  for (int i = 0; i < FCB_ENTRIES; i++) {
	  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i] = 0; 
  }
}



//check whether aim file name is correspond with file in storage
__device__ bool check_filename(char *n1, char *n2){
  while(*n1 == *n2){
    if(*n1 != '\0'){
      n1++;
      n2++;
    }
    else if(*n1 == '\0'){
      return true;
    }
  }
  return false;
}

//check whether file system is full. If full return -1, else return empty VCB number
__device__ int empty_in_FCB(FileSystem *fs){
  int ans=0;
  for (int i = 0; i < fs->FCB_ENTRIES; i++){
    if(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i] == 0){//等于0说明还有空
      ans++;
    }
  }
  return ans;
}

__device__ int check_filefull(FileSystem *fs){
  for (int i = 0; i < fs->FCB_ENTRIES; i++){
    if(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i] == 0){//等于0说明还有空
      return i;
    }
  }
  return -1;
}

__device__ bool found_file(u32 FCB_num){
  return FCB_num != -1;
}

__device__ bool check_validbit(FileSystem *fs, int FCB_num){
  return fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num] == 1;
}

__device__ bool check_blockaddr(int block_num){
  return (block_num >=0 && block_num <= 1023);
}
//------------------set funciton
__device__ void set_validbit(FileSystem *fs, int FCB_num){
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num] = 1;
}

__device__ void set_filename(FileSystem *fs, int FCB_num, char *s){
  int name_len =0;
  while(*s != '\0' && name_len<=19){
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +1 +name_len] = *s;
    s++;
    name_len++;
  }
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +1 +name_len] = '\0';
}

__device__ void set_size(FileSystem *fs, int FCB_num, u32 size ){
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +21] = size / 128;
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +22] = size % 128;
}
__device__ void set_createtime(FileSystem *fs, int FCB_num, u32 gtime){
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +24] = gtime / 128;
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +25] = gtime % 128;
}

__device__ void set_modifytime(FileSystem *fs, int FCB_num, u32 gtime){
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +26] = gtime / 128;
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +27] = gtime % 128;
}

__device__ void set_address(FileSystem *fs, int FCB_num, int block_num){
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +28] = block_num % 128;
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +29] = block_num / 128;
}

//-------------------------get function
__device__ int get_block_addr(FileSystem *fs, int FCB_num, int ADDRESS_BASE){
  int LOW_pos = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +28];
  int HIGH_pos = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +29];
  int temp_addr = LOW_pos * pow(ADDRESS_BASE, 0) + HIGH_pos * pow(ADDRESS_BASE, 1);
  return temp_addr;
}
__device__ int get_storage_addr(FileSystem *fs, int FCB_num, int ADDRESS_BASE){
  int LOW_pos = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +28];
  int HIGH_pos = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * FCB_num +29];
  int temp_addr = LOW_pos * pow(ADDRESS_BASE, 0) + HIGH_pos * pow(ADDRESS_BASE, 1); //得到在storage中的block number
  temp_addr *= fs->STORAGE_BLOCK_SIZE;
  int addr = temp_addr + fs->FILE_BASE_ADDRESS;
  return addr; //以byte为单位
}

__device__ int get_bitmap(FileSystem *fs, int block_num){//在bitmap中将这个bloc_num对应的bit取出来
  int row = block_num / 8;
	int column = block_num % 8;
  int tmp_num = fs->volume[row];
  return (tmp_num >> (8-column-1)) & 1;
}

__device__ int update_bitmap(FileSystem *fs, int block_num, int bit) {//更新这个block的bitmap为type类型（0/1）
  if(check_blockaddr(block_num)){
    int row = block_num / 8;
    int column = block_num % 8;//获取当前bit的位置

    if (bit == 1) {
      fs->volume[row] |= (1 << column); 
    }
    else {
      uchar change = ~(1 << column);
      fs->volume[row] &= change;
    }
  }
  return 0;
}

__device__ void do_compaction(FileSystem *fs, u32 fp){//fp是要删除的文件的pointer（block number）

  // printf("This is one compactions\n");
  int shift_addr = get_storage_addr(fs, fp, ADDRESS_BASE); //要删除文件的实际地址
  int shift_block = get_block_addr(fs, fp, ADDRESS_BASE); //要删除文件的block起始位置
  int fp_size =  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp +21] * ADDRESS_BASE + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp +22];
  int block_takeup = fp_size/fs->STORAGE_BLOCK_SIZE;
  if((fp_size % fs->STORAGE_BLOCK_SIZE)!=0) block_takeup++;
  // printf("This is blocktake up %d", block_takeup);
  //设置这个文件的bitmap为0
  for(int i=0; i<block_takeup; i++){
    update_bitmap(fs, shift_block + i, 0);
  }

  //检查这个文件之后有多少个block要移动
  int storage_block_tomove = 0;//storage中要move的block数量
  for(int i = shift_block + block_takeup; i< 1024; i++){
    if(get_bitmap(fs, i) == 1) storage_block_tomove ++;
  }


  //遍历所有的文件，如果在这个removefile之后，则更新其VCB，FCB，并且shift storage
  for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		if (i != fp && fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i] != 0){
      int cur_addr = get_storage_addr(fs, i, ADDRESS_BASE); //当前遍历文件的实际地址
      int cur_block = get_block_addr(fs, i, ADDRESS_BASE); //当前遍历文件的block起始位置     
      int cur_block_takeup =  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +21] * ADDRESS_BASE + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +22];
      //对比storage addr的大小
      if(cur_addr > shift_addr){
        //更新FCB中的addr
        int new_block = cur_block - shift_block;
        set_address(fs, i, new_block);
        //更新VCB
        //1.删除移动前的block bit为0
        for (int j = 0; j < cur_block_takeup; j++) {
					update_bitmap(fs, cur_block + j, 0);
				}
        //2.设置移动后的block bit为1
        for (int j = 0; j < cur_block_takeup; j++) {
					update_bitmap(fs, new_block + j, 1);
				}
        
      }
    }

  }

  //移动storage----
  int tmp_add = shift_addr;
  for (int i = 0; i <  storage_block_tomove; i++) {//要移动storage_block_tomove这么多次
    for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){//每一个block要移动size次
      fs->volume[tmp_add] = fs->volume[tmp_add + block_takeup * fs->STORAGE_BLOCK_SIZE];
      tmp_add++;
    }
  }

  //更新global storage pointer
  gstorage_ptr = gstorage_ptr - block_takeup * fs->STORAGE_BLOCK_SIZE;
}




__device__ void swap_FCB(FileSystem *fs, u32 fp1, u32 fp2){
  for (int i = 0; i < 32; i++){
		uchar temp = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp1 + i];
		fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp1 + i] = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp2 + i];
		fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp2 + i] = temp;
	}
}

__device__ void Display(uchar * s) {
	while (*s != '\0') {
		printf("%c", (char) *s);
		s++;
	}
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  gtime++;
  u32 FCB_num = -1;
  // printf("this is 154 valid bit in FCB---%d----",fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * 154]);
  // printf("There are %d empty sapce in FCB----", empty_in_FCB(fs));
  // printf("When open----The global storage ptr is%d--", gstorage_ptr);

  for(int i = 0; i < fs->FCB_ENTRIES; i++){
    if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i] != 0 && check_filename(s, (char *) &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +1]) ){
      FCB_num = i;
      break;
    }
  }

  if(found_file(FCB_num)) return FCB_num;

  // printf("Now is opening the %dth file----", FCB_num);

  if(op == G_WRITE){
    if(check_filefull(fs) == -1){
      printf("The file system is full!\n");
      return FCB_num;
    }
    else{
      int zerofile_FCB_pos = check_filefull(fs);

      //set file attribute
      // printf("This is zero file position--%d\n ",zerofile_FCB_pos );
      set_validbit(fs, zerofile_FCB_pos);
      set_filename(fs, zerofile_FCB_pos, s);
      set_size(fs,zerofile_FCB_pos, 0); //file size is zero
      set_createtime(fs, zerofile_FCB_pos, gtime);
      set_modifytime(fs, zerofile_FCB_pos, gtime);
      // printf("!!!!!!!AFTER OPEN 154 File this is 154 valid bit in FCB---%d----",fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * 154]);
      return zerofile_FCB_pos;
    }
  }

  return FCB_num;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  gtime++;
  if(!check_validbit(fs, fp)){
    printf("The file is not valid!\n");
    return ;
  }
  else{
    int storage_addr_ptr = get_storage_addr(fs, fp, ADDRESS_BASE);
    for (int i = 0; i < size; i++) {
      output[i] = fs->volume[storage_addr_ptr + i];
    }
  }

}





__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp){
  if( fp<0 || fp >= 1024){
    printf("The file is not valid!!!\n");
    return -1;
  }
  if (size >= fs->MAX_FILE_SIZE){
    printf("The file size exceeds the limit!!!\n");
    return -1;
  }

  gtime++;
  int oldfile_size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21] * ADDRESS_BASE + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22];
  if(oldfile_size != 0){//clear 要被覆写的文件 的VCB
    // printf("The %d file size is not zero and needs compaction\n", fp);
    //移除原有的文件并且做compaction
    do_compaction(fs,fp);
    set_size(fs, fp, 0);
  }
  // printf("!!!!!!!CHECK WRITE---154 File valid bit in FCB---%d----\n",fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * 154]);
  //allocate storage
  set_address(fs, fp, gstorage_ptr);//应该compaciton之后再改gstorage ptr
  
  int block_num = size / fs->STORAGE_BLOCK_SIZE;
  int remain_bit = size % fs->STORAGE_BLOCK_SIZE;
  if(remain_bit !=0) block_num ++;

  int start_addr = gstorage_ptr + fs->FILE_BASE_ADDRESS;
  int start_block = gstorage_ptr/ fs->STORAGE_BLOCK_SIZE; //不用减去filebase addr，因为直接从storage 0开始
  for(int j=0; j<block_num; j++){
    update_bitmap(fs, start_block + j, 1);
  }


  //写进storage
  for(int i=0; i<size; i++){
    fs->volume[start_addr++] = input[i];
  }

  //更新FCB
  // printf("!!!!!!!CHECK WRITE---154 File valid bit in FCB---%d----\n",fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * 154]);
  set_size(fs,fp,size);
  set_modifytime(fs, fp, gtime);

  //更新 gstorage_ptr
  gstorage_ptr += block_num * fs->STORAGE_BLOCK_SIZE;
  // printf("The size is%d------The global storage ptr is%d--",size, gstorage_ptr);
  // printf("!!!!!!!AFTER WRITE 154 File this is 154 valid bit in FCB---%d----",fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * 154]);
  return 0;
}




__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  gtime++;
  if(op == LS_D){//用bubble sort来排序
    for(int i=0; i< fs->FCB_ENTRIES -1; i++){
      for(int j=0; j< fs->FCB_ENTRIES -1-i; j++){
        if(check_validbit(fs, j) && check_validbit(fs,j+1)){
          int left_modified_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j +26] *128 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j +27];
          int right_modified_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * (j+1) +26] *128 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * (j+1) +27];
          if(left_modified_time < right_modified_time){
            swap_FCB(fs, j, j+1);//只需要调整FCB的顺序
          }
        }
      }
    }
    //按FCB顺序展示File name
    printf("===sort by modified time===\n");
    for(int i =0; i<fs->FCB_ENTRIES; i++){
      if(check_validbit(fs,i)){
        uchar* name = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +1];
        Display(name);
        printf("\n");
      }
    }


  }
  else{//用bubble sort来排序
    for(int i=0; i< fs->FCB_ENTRIES -1; i++){
      for(int j=0; j< fs->FCB_ENTRIES -1-i; j++){
        if(check_validbit(fs, j) && check_validbit(fs, j+1)){
          int left_size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j +21] *128 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j +22];
          int right_size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * (j+1) +21] *128 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * (j+1) +22];
          if(left_size < right_size){
            swap_FCB(fs, j, j+1);
          }
          else if(left_size == right_size){//若size相同，则按照create time 来排序
            int left_create_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j +24] *128 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j +25];
            int right_create_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * (j+1) +24] *128 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * (j+1) +25];
            if(left_create_time > right_create_time){
              swap_FCB(fs, j, j+1);//只需要调整FCB的顺序
            }
          }
        }
      }
    }

    //按FCB顺序展示File name
    printf("===sort by file size===\n");
    for(int i =0; i<fs->FCB_ENTRIES; i++){
      if(check_validbit(fs, i)){
        uchar* name = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +1];
        int size = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +21] *128 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +22];
        int create_time = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +24] *128 + fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +25];
        Display(name);
        printf(" %d",size);
        printf("\n");
      }
    }

  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  gtime++;
  if(op == RM){
    //在FCB中找到要remove的file
    int removefile_pos = -1;
    for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			if (check_filename(s, (char *) &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i +1])) {
				removefile_pos = i;
				break;
			}
		}

    //更新FCB---更改valid bit为0
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * removefile_pos] = 0;
    //移除文件并做compaction(包含更新移动了的file的VCB)
    do_compaction(fs, removefile_pos);

  }
}
