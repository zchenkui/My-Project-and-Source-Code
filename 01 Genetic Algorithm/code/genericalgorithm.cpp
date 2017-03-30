#include "geneticalgorithm.h"

/*this function is used by qsort function*/
int f_compare(void const *a, void const *b)
{
	return ((Chromosome *)a)->fitnessValue > ((Chromosome *)b)->fitnessValue ? -1 :
		(((Chromosome *)a)->fitnessValue < ((Chromosome *)b)->fitnessValue ? 1 : 0);
}

/*initialize chromosome list*/
void initialize(Chromosome *chromo_list, char const(*graph)[NODE_NUMBER])
{
	/*
	** generate candidate solution randomly and calculate fitness value for each chromosome.
	*/
	for (Chromosome *p_chromo = chromo_list; p_chromo < chromo_list + POP_SIZE; p_chromo++) {
		for (int i = 0; i < NODE_NUMBER; i++) {
			p_chromo->solution[i] = randi() % 3;
		}
		p_chromo->fitnessValue = fitness(graph, p_chromo->solution);
	}
}

/*calculate total fitness of population*/
double total_fitness(Chromosome const *chromo_list)
{
	double total_fitness = 0.0;
	for (unsigned i = 0; i < POP_SIZE; i++) {
		total_fitness += chromo_list[i].fitnessValue;
	}

	return total_fitness;
}

/*select a chromosome with*/
unsigned int roulette_selection(Chromosome const *chromo_list, double sum_fitness)
{
	unsigned int selected_index = 0;
	double accumulate = 0.0;
	double criteria = 0.0;

	criteria = randf() * sum_fitness;

	for (unsigned i = 0; i < POP_SIZE; i++) {
		accumulate += chromo_list[i].fitnessValue;
		if (accumulate >= criteria) {
			selected_index = i; break;
		}
	}

	return selected_index;
}

/*select a chromosome with tournament*/
unsigned int tournament_selection(Chromosome const *chromo_list) 
{
	unsigned int selected_chromo_indices[K_CANDIDATE] = { 0 };
	unsigned int best_index;
	double best_fitness;
	
	/*
	** select k candidate chromosomes from gene pool randomly. make sure they are different from each other.
	*/
	while (1) {
		int distinct = 1;
		
		for (int i = 0; i < K_CANDIDATE; i++) {
			selected_chromo_indices[i] = randi() % POP_SIZE;
		}

		/*
		** check if there are two same indices.
		*/
		for (int i = 0; i < K_CANDIDATE - 1; i++) {
			for (int j = i + 1; j < K_CANDIDATE; j++) {
				if (selected_chromo_indices[i] == selected_chromo_indices[j]) {
					distinct = 0; break;
				}
			}
			if (distinct == 0) break;
		}

		if (distinct == 1) break;	/*if all indices are different, break out while statement*/
	}

	/*
	** select the best chromosome in k candidates
	*/
	best_index = selected_chromo_indices[0];
	best_fitness = chromo_list[selected_chromo_indices[0]].fitnessValue;
	for (int i = 1; i < K_CANDIDATE; i++) {
		if (chromo_list[selected_chromo_indices[i]].fitnessValue > best_fitness) {
			best_index = selected_chromo_indices[i];
			best_fitness = chromo_list[selected_chromo_indices[i]].fitnessValue;
		}
	}

	return best_index;
}

/*
**crossover chromosomes and generate children population. you should choose a crossover method by macro.
**	1--point crossover
**	2--mask crossover
*/
void crossover(Chromosome const *parent_chromo_list, Chromosome *children_chromo_list)
{
	int crossover_position;
	Chromosome chromo;
	chromo.fitnessValue = 0.0;
	int mask[NODE_NUMBER] = { 0, };
	double sum_fitness = total_fitness(parent_chromo_list);

	/*generate mask*/
	for (int i = 0; i < NODE_NUMBER; i++) {
		mask[i] = randi() % 2;
	}

	/*
	** generate children population by crossover
	*/
	for (int i = 0; i < POP_SIZE / 2; i++) {
		unsigned int chromo_index_1 = 0;
		unsigned int chromo_index_2 = 0;

		/*
		** select two different chromosome frome parent chromosome list.
		*/
		while (1) {
			switch (SELECT_METHOD)
			{
			case 1:	/*roulette selection*/
				chromo_index_1 = roulette_selection(parent_chromo_list, sum_fitness);
				chromo_index_2 = roulette_selection(parent_chromo_list, sum_fitness);
				
				break;

			case 2:	/*tournament selection*/
				chromo_index_1 = tournament_selection(parent_chromo_list);
				chromo_index_2 = tournament_selection(parent_chromo_list);

				break;
			default:
				break;
			}
			/*
			** make sure that two indices are different.
			*/
			if (chromo_index_1 != chromo_index_2)
				break;
		}

		switch (CROSS_METHOD)
		{
		case 1:	/*point crossover*/

			/*choose a crossover point*/
			while (1) {
				crossover_position = randi() % NODE_NUMBER;
				if (crossover_position != 0 && crossover_position != NODE_NUMBER - 1)
					break;
			}

			for (int j = 0; j < NODE_NUMBER; j++) {
				if (j < crossover_position) {
					children_chromo_list[2 * i].solution[j] = parent_chromo_list[chromo_index_1].solution[j];
					children_chromo_list[2 * i + 1].solution[j] = parent_chromo_list[chromo_index_2].solution[j];
				}
				else {
					children_chromo_list[2 * i].solution[j] = parent_chromo_list[chromo_index_2].solution[j];
					children_chromo_list[2 * i + 1].solution[j] = parent_chromo_list[chromo_index_1].solution[j];
				}
			}

			break;

		case 2:	/*mask crossover*/

			for (int j = 0; j < NODE_NUMBER; j++) {			
				if (mask[j] == 0) {	/*if mask == 0, ...*/
					children_chromo_list[2 * i].solution[j] = parent_chromo_list[chromo_index_1].solution[j];
					children_chromo_list[2 * i + 1].solution[j] = parent_chromo_list[chromo_index_2].solution[j];
				}
				else {	/*if mask == 1, ...*/
					children_chromo_list[2 * i].solution[j] = parent_chromo_list[chromo_index_2].solution[j];
					children_chromo_list[2 * i + 1].solution[j] = parent_chromo_list[chromo_index_1].solution[j];
				}
			}

			break;

		default:
			break;
		}	/*end of switch (CROSS_METHOD)*/

	}	/*end of for (int i = 0; i < POP_SIZE / 2; i++)*/
}



/*mutate chromosome to a new type*/
void mutation(Chromosome *chromo_list, double m_rate)
{
	char new_color = 0;
	for (Chromosome *p_chromo = chromo_list; p_chromo < chromo_list + POP_SIZE; p_chromo++) {
		for (int i = 0; i < NODE_NUMBER; i++) {
			if (randf() <= m_rate) {
				/*
				** we should select a new color which is different from the current one.
				*/
				while ((new_color = randi() % 3) == p_chromo->solution[i])
					;
				p_chromo->solution[i] = new_color;
			}
		}
	}
}

/*select the best solution in the current population*/
unsigned select_elite(Chromosome const *chromo_list)
{
	double best_fit = 0.0;
	unsigned int best_index = 0;

	for (unsigned i = 0; i < POP_SIZE; i++) {
		if (chromo_list[i].fitnessValue > best_fit) {
			best_fit = chromo_list[i].fitnessValue;
			best_index = i;
		}
	}

	/*return index of the current best*/
	return best_index;
}

/*assessment strategy is a local search algorithm. hybrid number: 1*/
int assessment_strategy(char const (*graph)[NODE_NUMBER], Chromosome *chromo)
{
	Chromosome tmp_chromo = *chromo;	/*by using temp firefly, the current firefly will not be affected*/
	int flag_change = 0;	/*fitness has been improved (1) or not (0)*/
	int eval_times = 0;	/*how many times to calculate the fitness*/
	Conflict_Infor conflict_infor;	/*conflict list is used to record the conflict number for each node*/
	int max_conflict_index = 0;

	/*
	** find the max conflict node.
	*/
	max_conflict_index = solution_conflict(graph, tmp_chromo.solution, &conflict_infor);

	/*
	** convert the max conflict node to some other color, and check whether the fitness value is improved or not.
	*/
	for (char color = 0; color < 3; color++) {
		if (color != chromo->solution[max_conflict_index]) {
			tmp_chromo.solution[max_conflict_index] = color;
			tmp_chromo.fitnessValue = fitness(graph, tmp_chromo.solution);
			eval_times += 1;
			if (tmp_chromo.fitnessValue >= chromo->fitnessValue) {
				flag_change = 1; break;
			}
		}
	}

	/*
	** update current firefly if found a better solution.
	*/
	if (flag_change == 1) {
		*chromo = tmp_chromo;
	}

	/*return times of calling fitness function.*/
	return eval_times;
}


/*hill climbing is a local search algorithm. hybrid number: 2*/
int hill_climbing(char const (*graph)[NODE_NUMBER], Chromosome *current_chromo)
{
	Chromosome tmp_chromo = *current_chromo;	/*by using temp chromosome, the current chromosome will not be affected*/
	int eval_times = 0;	/*how many times to calculate the fitness*/
	Conflict_Infor conflict_infor;	/*conflict infor is used to record the conflict number for each node*/
	int max_conflict_index = 0;
	int selected_index = 0;
	int flag_found = 0;

	int count = 0;
	while (count < MAX_HILLCLIMB) {

		/*
		** compute conflict list, see problem.cpp for details
		*/
		max_conflict_index = solution_conflict(graph, tmp_chromo.solution, &conflict_infor);

		/*if no conflict, we found a solution*/
		if (conflict_infor.len == 0) {
			flag_found = 1;
			break;	/*break while*/
		}

		/*
		** randomly select a node whose conflict is not 0
		*/
		selected_index = conflict_infor.conflict_nodes[randi() % conflict_infor.len];

		/*
		** convert the current selected node to some other color, and check whether the fitness value is improved or not.
		*/
		for (char color = 0; color < 3; color++) {
			if (color != current_chromo->solution[selected_index]) {
				tmp_chromo.solution[selected_index] = color;
				tmp_chromo.fitnessValue = fitness(graph, tmp_chromo.solution);
				eval_times += 1;

				/*
				** if new fitness if better than old one, update current firefly
				*/
				if (tmp_chromo.fitnessValue > current_chromo->fitnessValue) {
					current_chromo->solution[selected_index] = tmp_chromo.solution[selected_index];
					current_chromo->fitnessValue = tmp_chromo.fitnessValue;
				}

				if (tmp_chromo.fitnessValue == 1.0) {
					flag_found = 1;
					break;	/*break for*/
				}
			}
		}

		if (flag_found == 1) break;	/*break while*/

		count += 1;
	}

	return eval_times;
}

/*scaling fitness*/
void scaling(Chromosome *chromo_list)
{
	double max_fitness = 0.0;
	double min_fitness = 1.0;

	for (int i = 0; i < POP_SIZE; i++) {
		if (chromo_list[i].fitnessValue > max_fitness) {
			max_fitness = chromo_list[i].fitnessValue;
		}
		if (chromo_list[i].fitnessValue < min_fitness) {
			min_fitness = chromo_list[i].fitnessValue;
		}
	}

	if (min_fitness != max_fitness) {
		for (int i = 0; i < POP_SIZE; i++) {
			double current_fitness = chromo_list[i].fitnessValue;
			chromo_list[i].fitnessValue = (current_fitness - min_fitness) / (max_fitness - min_fitness);
		}
	}
}

/*calculate elapsed times, convert it to string*/
void elapsed_times(time_t const *start_time, time_t const *end_time, char *used_time)
{
	long elapsed_seconds = (long)difftime(*end_time, *start_time);
	int seconds;
	int minutes;
	int hours;

	/*
	** calculate hours, minutes and secondes.
	*/
	seconds = elapsed_seconds % 60;
	minutes = elapsed_seconds / 60;
	hours = minutes / 60;
	minutes = minutes % 60;

	sprintf(used_time, "%5d hour(s) %4d minute(s) %4d second(s)", hours, minutes, seconds);
}

/*genetic algorithm*/
Result *genetic_algorithm(char const (*graph)[NODE_NUMBER])
{
	Chromosome parents[POP_SIZE];
	Chromosome children[POP_SIZE];

	unsigned int parent_best = 0;	/*the index of current best chromosome*/

	Result *result_record = (Result *)malloc(sizeof Result);
	if (result_record == NULL) {
		printf("cannot allocate memory\n");
		exit(EXIT_FAILURE);
	}

	double gbest = 0.0;	/*the global best fitness*/
	int count = 0;
	double eval_times = 0.0;	/*evaluation times of object function*/
	double gbest_list[MAX_LOOP] = { 0.0 };	/*record all global best fitness value*/
	char current_best_solution[NODE_NUMBER];
	int success = 0;

	char *s_start_time = NULL;
	char *s_end_time = NULL;
	time_t start_time;
	time_t end_time;
	double used_time = 0.0;
	char s_used_time[50] = { 0 };
	char s_elapsed_times[100] = "";

	/*
	** get start time.
	*/
	start_time = time(NULL);
	s_start_time = ctime(&start_time);

	/*
	** initialize firefly list, then set old fire fly list.
	*/
	initialize(parents, graph);
	memset(children, 0, sizeof children);
	parent_best = select_elite(parents);
	gbest = parents[parent_best].fitnessValue;
	gbest_list[0] = gbest;

	while (count < MAX_LOOP) {
		/*
		** if the best solution is found.
		*/
		if (gbest == 1.0) {
			success = 1;

			if (PRINT_DETAIL) {
				printf("\tsolution found\n\n");
			}
			break;
		}

		if (USE_SCALING) {
			scaling(parents);
		}

		/*
		** crossover and mutation
		*/
		crossover(parents, children);
		mutation(children, MUTATE_RATE);

		/*
		** keep parents' elite
		*/
		if (USE_ELITE) {
			children[parent_best] = parents[parent_best];
		}

		/*
		** calculate fitness
		*/
		for (int i = 0; i < POP_SIZE; i++) {
			children[i].fitnessValue = fitness(graph, children[i].solution);
			eval_times += 1;
		}

		/*
		** update parents
		*/
		memcpy(parents, children, sizeof children);
		parent_best = select_elite(parents);
		memset(children, 0, sizeof children);

		/*
		** use hybrid
		*/
		if (parents[parent_best].fitnessValue != 1.0 && USE_HYBRID) {
			switch (HYBRID)
			{
			case 1:
				eval_times += assessment_strategy(graph, parents + parent_best);
				break;
			case 2:
				eval_times += hill_climbing(graph, parents + parent_best);
				break;
			default:
				break;
			}
		}

		memcpy(current_best_solution, parents[parent_best].solution, sizeof parents->solution);
		gbest = parents[parent_best].fitnessValue;
		gbest_list[count] = gbest;

		if (PRINT_DETAIL) {
			printf("\tLoop %4d ==========> %.5f\n", count + 1, gbest);
		}

		count += 1;
	} /*end of while*/

	/*
	** calculate elapsed times.
	*/
	end_time = time(NULL);
	s_end_time = ctime(&end_time);
	elapsed_times(&start_time, &end_time, s_elapsed_times);

	/*
	** save result to record.
	*/
	result_record->success = success;
	result_record->eval_times = eval_times;
	result_record->loop_times = count;
	memcpy(result_record->solution, current_best_solution, sizeof current_best_solution);
	memcpy(result_record->gbest_list, gbest_list, sizeof gbest_list);
	strcpy(result_record->start_time, s_start_time);
	strcpy(result_record->end_time, s_end_time);
	strcpy(result_record->s_elapsed_times, s_elapsed_times);

	return result_record;

	return NULL;
}

/*save result to two files*/
int save_result(Result const *result, char const *file_name)
{
	FILE *file_csv = NULL;	/*save global best fitness value list*/
	FILE *file_txt = NULL;	/*save other information*/

	char const *csv = ".csv";
	char const *txt = ".txt";
	char csvfile_name[200] = "";
	char txtfile_name[200] = "";

	/*
	** generate file name
	*/
	strcat(csvfile_name, file_name);
	strcat(csvfile_name, csv);
	strcat(txtfile_name, file_name);
	strcat(txtfile_name, txt);

	if ((file_csv = fopen(csvfile_name, "w")) == NULL) {
		printf("[FIREFLYALGORITHM.cpp--save_result--ERROR] cannot open csv file\n");
		exit(EXIT_FAILURE);
	}
	if ((file_txt = fopen(txtfile_name, "w")) == NULL) {
		printf("[FIREFLYALGORITHM.cpp--save_result--ERROR] cannot open txt file\n");
		exit(EXIT_FAILURE);
	}

	/*
	**save global best list to csv file
	*/
	for (int i = 0; i < result->loop_times; i++) {
		fprintf(file_csv, "%8d, %.8f\n", i + 1, result->gbest_list[i]);
	}

	/*
	**save other information to txt file
	*/
	fprintf(file_txt, "Start time: \t %s\n", result->start_time);
	fprintf(file_txt, "Find optimal value or not: \t %d\n", result->success);
	fprintf(file_txt, "Loop times: \t %d\n", result->loop_times);
	fprintf(file_txt, "Evaluation times: \t %.9e\n", result->eval_times);
	fprintf(file_txt, "Used times: \t %s\n", result->s_elapsed_times);
	fprintf(file_txt, "The best solution is: \t \n");
	for (int i = 0; i < NODE_NUMBER; i++) {
		fprintf(file_txt, "%3d, ", result->solution[i]);
	}
	fprintf(file_txt, "\n\n");
	fprintf(file_txt, "End time: \t %s\n", result->end_time);

	fclose(file_csv);
	fclose(file_txt);

	return EXIT_SUCCESS;
}