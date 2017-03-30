#ifndef _HEADER_GA_H
#define _HEADER_GA_H	1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mt.h"
#include "problem.h"

#define POP_SIZE	200
#define MAX_LOOP	10000
#define MAX_HILLCLIMB	45
#define MUTATE_RATE	0.014
#define USE_ELITE	1
#define USE_SCALING	1
#define CROSS_METHOD	2	/*crossover method.	1--point crossover.	2--mask crossover*/
#define SELECT_METHOD	2	/*select method. 1--roulette select. 2--tournament select*/
#define K_CANDIDATE	2	/*number of candidate chromosome in tournament select*/
#define USE_HYBRID	0
#define HYBRID		2
#define PRINT_DETAIL	0

/*chromosome structure*/
typedef struct Chromosome {
	char solution[NODE_NUMBER];	/*candidate solution*/
	double fitnessValue;	/*fitness of candidate solution*/
} Chromosome;

/*record the result*/
typedef struct Result {
	int success;
	int loop_times;
	double eval_times;
	double gbest_list[MAX_LOOP];
	char solution[NODE_NUMBER];
	char start_time[50];
	char end_time[50];
	char s_elapsed_times[100];
} Result;

/*this function is used by qsort function*/
int f_compare(void const *a, void const *b);

/*initialize chromosome list*/
void initialize(Chromosome *chromo_list, char const(*graph)[NODE_NUMBER]);

/*calculate total fitness of population*/
double total_fitness(Chromosome const *chromo_list);

/*select a chromosome with roulette*/
unsigned int roulette_selection(Chromosome const *chromo_list, double sum_fitness);

/*select a chromosome with tournament*/
unsigned int tournament_selection(Chromosome const *chromo_list);

/*
**crossover chromosomes and generate children population. you should choose a crossover method by macro.
**	1--point crossover
**	2--mask crossover
*/
void crossover(Chromosome const *parent_chromo_list, Chromosome *children_chromo_list);

/*mutate chromosome to a new type*/
void mutation(Chromosome *chromo, double m_rate);

/*select the best solution in the current population*/
unsigned select_elite(Chromosome const *chromo_list);

/*scaling fitness*/
void scaling(Chromosome *chromo_list);

/*assessment strategy is a local search algorithm. hybrid number: 1*/
int assessment_strategy(char const (*graph)[NODE_NUMBER], Chromosome *chromo);

/*hill climbing is a local search algorithm. hybrid number: 2*/
int hill_climbing(char const (*graph)[NODE_NUMBER], Chromosome *current_chromo);

/*calculate elapsed times, convert it to string*/
void elapsed_times(time_t const *start_time, time_t const *end_time, char *used_time);

/*genetic algorithm*/
Result *genetic_algorithm(char const (*graph)[NODE_NUMBER]);

/*save result to two files*/
int save_result(Result const *result, char const *file_name);

#endif