#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mt.h"
#include "problem.h"
#include "geneticalgorithm.h"

#define MAX_RUN	30
#define D_NUM	11	/*length of d list*/
#define SAVE_GRAPH	0	/*save graph or not*/
#define SAVE_RESULTS	1	/*save results or not*/
#define GRAPH_SAVE_PATH	"..\\graph\\"
#define RESULTS_SAVE_PATH	"..\\results\\"
#define FINAL_RESULT_PATH	"..\\final results\\"

/*generate full record save path*/
void generate_save_path(char *save_path, char const *save_directory, char const *file_name);

int main()
{
	setseed((unsigned)time(NULL));

	char graph[NODE_NUMBER][NODE_NUMBER];
	float d_list[D_NUM] = { 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	char *s_d_list[] = { " d_15 ", " d_20 ", " d_25 ", " d_30 ", " d_40 ", " d_50 ",
		" d_60 ", " d_70 ", " d_80 ", " d_90 ", " d_100 ", };
	int sr = 0;
	double avg_eval_times = 0.0;
	int sr_list[D_NUM] = { 0 };
	double avg_eval_list[D_NUM] = { 0.0 };
	char full_path[200] = "";
	char file_name[100] = "";
	FILE *final_result = NULL;

	/*
	** print start time.
	*/
	time_t current_time = time(NULL);
	printf("Start---%s", ctime(&current_time));

	/*
	** for each d ...
	*/
	for (int i = 0; i < D_NUM; i++) {
		float d = d_list[i];
		avg_eval_times = 0.0;
		sr = 0;

		printf("d = %f begin:\n", d);

		/*
		** for each try ...
		*/
		for (int k = 0; k < MAX_RUN; k++) {
			/*
			** generate random graph
			*/
			generate_random_graph(graph, d);
			memset(full_path, 0, sizeof full_path);
			memset(file_name, 0, sizeof file_name);

			/*
			** if save graph ...
			*/
			if (SAVE_GRAPH) {
				strcpy(file_name, "graph90");
				strcat(file_name, s_d_list[i]);
				generate_save_path(full_path, GRAPH_SAVE_PATH, file_name);
				save_graph(graph, full_path, ".csv");
				memset(full_path, 0, sizeof full_path);
				memset(file_name, 0, sizeof file_name);
			}

			/*
			** run genetic algorithm
			*/
			Result *p_result = genetic_algorithm(graph);

			/*
			** print result
			*/
			if (p_result->success) {
				printf("\t graph %3d ============> success\n", k);
				sr += 1;
				avg_eval_times += p_result->eval_times;
			}
			else {
				printf("\t graph %3d ============> fail\n", k);
			}

			if (SAVE_RESULTS) {
				strcpy(file_name, "result90");
				strcat(file_name, s_d_list[i]);
				generate_save_path(full_path, RESULTS_SAVE_PATH, file_name);
				save_result(p_result, full_path);
				memset(full_path, 0, sizeof full_path);
				memset(file_name, 0, sizeof file_name);
			}

			/*
			** ATTENTION: do NOT forget free malloc memory!
			*/
			free(p_result);
		}

		if (sr > 0) {
			sr_list[i] = sr;
			avg_eval_list[i] = avg_eval_times / sr;

			printf("d = %f finished. success: %d times, average evaluation times: %.6e\n\n", d, sr, avg_eval_list[i]);
		}
	}

	printf("all finish!\n\n");

	/*
	** print finish time
	*/
	current_time = time(NULL);
	printf("End---%s", ctime(&current_time));

	/*
	** save final result to csv file.
	*/
	strcpy(file_name, "final result 90");
	generate_save_path(full_path, FINAL_RESULT_PATH, file_name);
	strcat(full_path, ".csv");

	if ((final_result = fopen(full_path, "w")) == NULL) {
		printf("[MAIN.cpp--save_final_result--ERROR] cannot open file\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < D_NUM; i++) {
		fprintf(final_result, "%f, %d, %.6e\n", d_list[i], sr_list[i], avg_eval_list[i]);
	}

	fclose(final_result);

	return EXIT_SUCCESS;
}

/*generate full record save path*/
void generate_save_path(char *save_path, char const *save_directory, char const *file_name)
{
	time_t current_time = time(NULL);
	char time_string[50] = "";
	char *p_colon = NULL;

	strcpy(time_string, ctime(&current_time));
	/*because time string is ended by "\n\0", we must remove the '\n'*/
	time_string[strlen(time_string) - 1] = '\0';
	/*colon (':') will be replaced by '-', otherwise the fopen function will fail*/
	p_colon = strchr(time_string, ':');
	while (p_colon != NULL) {
		*p_colon = '-';
		p_colon = strchr(p_colon, ':');
	}
	/*
	** finally, the default file name will be something like: "Mon Jan 09 17-29-21 2017"
	*/

	/*
	** create full save path
	*/
	strcpy(save_path, save_directory);
	strcat(save_path, file_name);
	strcat(save_path, time_string);
}