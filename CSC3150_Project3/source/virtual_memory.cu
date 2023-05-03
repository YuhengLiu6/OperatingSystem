#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;//4-8KB store the VPN
    //initialize frequency number (8-12KB)
    vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] = 0;
  }
}

__device__ void init_swaptable(VirtualMemory *vm) {
  for(int i = 0; i < 5120; i++) {
    vm->swaptable[i] = 0;
  }
}


__device__ u32 storage_swaptoPM;

__device__ void Swap(VirtualMemory *vm, int frame_to_swap, u32 virtual_page_number, bool to_disk){
  if (to_disk == true){//use boolean to change whether swap into disk or from disk to pm
    //if swap into disk, update the storage index
    int tepVPN = vm->invert_page_table[frame_to_swap + vm->PAGE_ENTRIES];
    vm->swaptable[tepVPN] = (*vm->storage_index_ptr);
  }
  else{
    storage_swaptoPM = vm->swaptable[virtual_page_number];//swap到PM
    (*vm->storage_index_ptr) = storage_swaptoPM;
  }  
}



__device__ int check_hit(VirtualMemory *vm, int virtual_page_number){//check whether page table hit
  int frame = -1;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++){
    if ((vm->invert_page_table[i] != 0x80000000) && (vm->invert_page_table[i + vm->PAGE_ENTRIES] == virtual_page_number)){//valid bit Is VALID
        frame = i;
        break;
      }
    }

  return frame;
}

__device__ int check_empty(VirtualMemory *vm){
  int empty = -1;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++){
    if(vm->invert_page_table[i] == 0x80000000){
      empty = i;
      break;
    }
  }
  return empty;
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage, u32 *swaptable,
                        u32 *invert_page_table, int *pagefault_num_ptr,  int *storage_index_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {

  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->swaptable = swaptable;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;
  vm->storage_index_ptr = storage_index_ptr;
  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
  init_swaptable(vm);//intialize swap table
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  // return 123; //TODO
  uchar ans;

  int frame_number = -1;
  int empty_number = -1;
  int hit_time = 0;
  u32 phy_addr;
  u32 VPN = addr / vm->PAGESIZE;
  u32 REM = addr % vm->PAGESIZE; //the remaining data in last page
  
//If page table hit, that is, the read content is in physical memory
  frame_number = check_hit(vm, VPN);
  empty_number = check_empty(vm);

//case1:if page table hit
if(frame_number != -1){
  phy_addr = REM + frame_number * vm->PAGESIZE;
  //将所有的frequncy number加1，hit的设为0
  for(int i = 0; i < vm->PAGE_ENTRIES; i++){
    if(vm->invert_page_table[i]!= 0x80000000){
      vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
    }
  }
  vm->invert_page_table[frame_number + vm->PAGE_ENTRIES * 2] = 0;
  ans = vm->buffer[phy_addr];

  return ans;
  // printf("hit times is %d\n", ++hit_time);
}
//case2:if page not hit
else{
  //check whether page table has empty space
    //inner case 1----the page table is not full
  if(empty_number != -1){
    (*vm->pagefault_num_ptr)++;
    vm->invert_page_table[empty_number] = 0x7fffffff; //change valid bit
    vm->invert_page_table[empty_number + vm->PAGE_ENTRIES] = VPN; //update page table
    vm->invert_page_table[empty_number + vm->PAGE_ENTRIES * 2] = 0;

    phy_addr = REM + empty_number * vm->PAGESIZE;

    //swap the page from disk to physical memory
    Swap(vm, 0,VPN,false);

    for(int i=0; i< vm->PAGESIZE; i++){
      vm->buffer[empty_number * vm->PAGESIZE + i] = vm->storage[storage_swaptoPM + i];
    }
    ans = vm->buffer[phy_addr];
  }

  else{//inner case 2:----the page table is full
    (*vm->pagefault_num_ptr)++;
    // printf("This is VPN %d\n", VPN);
    u32 largest = 0;
    int frame_swap =0;
    for(int i=0;i< vm->PAGE_ENTRIES; i++){//traverse to find the one tha thas largest frenquency numebr and replace it
      if(vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] >= largest){
        largest = vm->invert_page_table[i + vm->PAGE_ENTRIES * 2];
        frame_swap = i;
      }
    }//get the largest one

    

    Swap(vm,frame_swap, 0,true);


    //SWAP largest frequency page back to disk-------
    for (int i = 0; i < vm->PAGESIZE; i++) {
      vm->storage[(*vm->storage_index_ptr) ++] = vm->buffer[frame_swap * vm->PAGESIZE + i]; 
    }

  
 
    //SWAP the page from disk to physical memory

    Swap(vm, 0, VPN, false);

    vm->invert_page_table[frame_swap + vm->PAGE_ENTRIES] = VPN; //update page table
    for(int i=0; i< vm->PAGESIZE; i++){
      vm->buffer[frame_swap * vm->PAGESIZE + i] = vm->storage[storage_swaptoPM + i];
    }

    phy_addr = REM + frame_swap * vm->PAGESIZE;

    for(int i = 0; i < vm->PAGE_ENTRIES; i++){
      if(vm->invert_page_table[i]!= 0x80000000){
        vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
        }
    }
    
    vm->invert_page_table[frame_swap + vm->PAGE_ENTRIES * 2 ] = 0; //设置新的frequency numebr为0
    ans = vm->buffer[phy_addr];
    }

    return ans;
}

}






__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  int frame_number = -1;
  int empty_number = -1 ;
  int frame_swap;
  // int frequency_number; //To record the used time of page in page table, the largest one will be replaced
  u32 phy_addr;
  u32 VPN = addr / vm->PAGESIZE;
  u32 REM = addr % vm->PAGESIZE; //the remaining data in last page

  //Go through Page Table
  frame_number = check_hit(vm, VPN);
  empty_number = check_empty(vm);

  //case1:if the page hit
  if(frame_number != -1){
    phy_addr = REM + frame_number * vm->PAGESIZE;
    vm->buffer[phy_addr] = value;

    for(int i = 0; i < vm->PAGE_ENTRIES; i++){
      if(vm->invert_page_table[i]!= 0x80000000){
        vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
      }
   }
    vm->invert_page_table[frame_number + vm->PAGE_ENTRIES * 2] = 0;
  }
  else{//case2:page table not hit
    
    (*vm->pagefault_num_ptr)++;
    //inner case 1: page table is not full
    if (empty_number != -1) {
			vm->invert_page_table[empty_number] = 0x7fffffff; //update valid bit
      vm->invert_page_table[empty_number + vm->PAGE_ENTRIES] = VPN; //update page table
      phy_addr = REM + empty_number * vm->PAGESIZE;

      for(int i = 0; i < vm->PAGE_ENTRIES; i++){
        if(vm->invert_page_table[i]!= 0x80000000){
          vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
        }
      }

      vm->invert_page_table[empty_number + vm->PAGE_ENTRIES * 2 ] = 0;
      vm->buffer[phy_addr] = value;
		}
    else{//inner case2: page table is full, need to swap LRU

      // printf("THis is write inner case 2\n");
      u32 largest = vm->invert_page_table[vm->PAGE_ENTRIES * 2];

      for(int i=0;i < vm->PAGE_ENTRIES; i++){//traverse to find the one tha thas largest frenquency numebr and replace it
        if(vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] >= largest){
          largest = vm->invert_page_table[i + vm->PAGE_ENTRIES * 2];
          frame_swap = i;
        }
      }//get the largest one
      
      Swap(vm, frame_swap, 0, true);
    //update page table
      for(int i=0;i < vm->PAGESIZE; i++){//write into disk
        vm->storage[(*vm->storage_index_ptr) ++] = vm->buffer[frame_swap * vm->PAGESIZE + i];
      }
      // printf("this is (*storage_index_ptr) %d", (*vm->storage_index_ptr));
      vm->invert_page_table[frame_swap + vm->PAGE_ENTRIES] = VPN;

      
      for(int i = 0; i < vm->PAGE_ENTRIES; i++){
        if(vm->invert_page_table[i]!= 0x80000000){
          vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
        }
      }
      vm->invert_page_table[frame_swap + vm->PAGE_ENTRIES * 2] = 0; //设置新的frequency numebr为0
      
      phy_addr = REM + frame_swap * vm->PAGESIZE;
      vm->buffer[phy_addr] = value;
    }
  }
}



__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i = 0; i < input_size; i++) {
        results[i] = vm_read(vm, i+ offset);
    }
}

