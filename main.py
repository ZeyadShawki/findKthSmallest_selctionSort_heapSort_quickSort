from numpy import random
import time
import pandas as pd
from functools import partial
import timeit
import numpy as np
import matplotlib.pyplot as plt

arraySizeTest = 100

def generateRandomArr(arrSize):
    x=random.randint(1000, size=(arrSize))
    return x


 

# =======================Quick Sort================================#
def partition(array, start, end):
    randpivot = random.randint(start, end)
    array[start], array[randpivot] = array[randpivot], array[start]
    pivot = start
    for i in range(start + 1, end + 1):
        if array[i] <= array[start]:
            pivot += 1
            array[i], array[pivot] = array[pivot], array[i]
    array[pivot], array[start] = array[start], array[pivot]
    return pivot


def quicksort(array, start=0):
    end = len(array) - 1

    def quicksort2(array, start, end):
        if start < end:
            pivot = partition(array, start, end)
            quicksort2(array, start, pivot - 1)
            quicksort2(array, pivot + 1, end)
        else:
            return

    return quicksort2(array, start, end)



def insertionsort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
            j-=1
        arr[j+1]=key
 
def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root

    if l < n and arr[i] < arr[l]:
        largest = l

    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])

        heapify(arr, n, largest)

 

def heapSort(arr):
    n = len(arr)

    # Build a maxheap.
    # Since last parent will be at ((n//2)-1) we can start at that location.

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements

    for i in range(n - 1, 0, -1):
        (arr[i], arr[0]) = (arr[0], arr[i])  # swap
        heapify(arr, i, 0)


 


def selection_sort(array,start,end):
 
    size = end - start + 1
    for i in range(start, end + 1):
        min = i
        for j in range(i + 1, end + 1):
            if array[min] < array[j]:
                min = j
        if min != i:
            (array[i], array[min]) = (array[min], array[i]) 


def merge(arr, l, m, r):

    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)

    for i in range(0, n1):
        L[i] = arr[l + i]

    for j in range(0, n2):
        R[j] = arr[m + 1 + j]

    i = 0
    j = 0
    k = l

    while i < n1 and j < n2:
        if L[i] >= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1


def mergeSorHybrid(arr, l, r, threshold):
    if l < r:
        m = l + (r - l) // 2
        if (r - l + 1 <= threshold):
           
            selection_sort(arr, l, r)
        else:
            mergeSorHybrid(arr, l, m, threshold)
            mergeSorHybrid(arr, m + 1, r, threshold)
            merge(arr, l, m, r)


def mergeSort(arr, l, r):
    if l < r:
        m = l + (r - l) // 2

        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
        merge(arr, l, m, r)

 




def findKthSmallest(array, k):
    def select(start, end, k_smallest):
        if start == end:
            return array[start]

        pivot_index = partition(array, start, end)
        if k_smallest == pivot_index:
            return array[k_smallest]
        elif k_smallest < pivot_index:
            return select(start, pivot_index - 1, k_smallest)
        else:
            return select(pivot_index + 1, end, k_smallest)

    return select(0, len(array) - 1, k - 1)






arr = generateRandomArr(arraySizeTest)
testArr = arr.copy()
st = time.time()
quicksort(testArr)
et = time.time()
print("Quick Sort With Size " + str(arraySizeTest) + " took") 
print((et - st) * 1000)

# print(arr)
 

testArr = arr.copy()
st = time.time()
insertionsort(testArr)
et = time.time()
print("Insertion Sort With Size " + str(arraySizeTest) + " took") 
print((et - st) * 1000)

# print(arr)
 

testArr = arr.copy()
# arr = generateRandomArr(arraySizeTest)
st = time.time()
heapSort(testArr)
et = time.time()
print("Heap Sort Sort With Size " + str(arraySizeTest) + " took") 
print((et - st) * 1000)

# print(arr)

testArr = arr.copy()
# arr = generateRandomArr(arraySizeTest)
st = time.time()
selection_sort(testArr, 0, len(arr) - 1)
et = time.time()
print("selection sort  With Size " + str(arraySizeTest) + " took") 
print((et - st) * 1000)
# print(testArr)
 
testArr = arr.copy()
st = time.time()
mergeSort(testArr, 0, len(arr) - 1)
et = time.time()
print("selection sort  With Size " + str(arraySizeTest) + " took") 
print((et - st) * 1000)

# print(testArr)
 