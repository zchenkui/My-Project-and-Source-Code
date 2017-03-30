#ifndef _HEADER_PROBLEM_H
#define _HEADER_PROBLEM_H	1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mt.h"

#define NODE_NUMBER	90	/*number of graph nodes*/

/*give the conflict information of current solution*/
typedef struct graph_conflict_list {
	int conflict_nodes[NODE_NUMBER];	/*record the conflict indices (conflict nodes)*/
	int conflict_numbers[NODE_NUMBER];	/*record the conflict number of each conflict node*/
	int len;	/*the number of conflict nodes*/
	int max_conflict_node;	/*max conflict node*/
	int max_conflict;	/*max conflict*/
}Conflict_Infor;

/*Given the node number and constraint density d, generate a random graph*/
void generate_random_graph(char(*graph)[NODE_NUMBER], float d);

/*save graph to a file, if the file name or extension is set to NULL, they will be set to default values*/
int save_graph(char const (*graph)[NODE_NUMBER], char const *filename, char const *extension);

/*calculate the fitness of a solution*/
double fitness(char const (*graph)[NODE_NUMBER], char const *solution);

/*given a graph and a solution, the conflict matrix is calculated.
conflict matrix is a matrix in which each row identifies those nodes whose colors conflict with current node.*/
void generate_conflict_matrix(char const (*graph)[NODE_NUMBER], char const *solution, char(*conflict_matrix)[NODE_NUMBER]);

/*for each node, calculate the number of conflict to other nodes, and return the "most conflict" node*/
int solution_conflict(char const (*graph)[NODE_NUMBER], char const *solution, Conflict_Infor *conflict_infor);

#endif