#include "problem.h"
#include "mt.h"

/*Given the node number and constraint density d, generate a random graph*/
void generate_random_graph(char(*graph)[NODE_NUMBER], float d)
{
	unsigned int total_links = (unsigned int)(NODE_NUMBER * d);	/*the total number of links in graph*/
	char part1[NODE_NUMBER / 3][NODE_NUMBER / 3] = { { 0 } };	/*subgraph 1*/
	char part2[NODE_NUMBER / 3][NODE_NUMBER / 3] = { { 0 } };	/*subgraph 2*/
	char part3[NODE_NUMBER / 3][NODE_NUMBER / 3] = { { 0 } };	/*subgraph 3*/

	int k = NODE_NUMBER / 3;	/*the node number of each subgraph*/
	unsigned int current_links = 0; /*the current number of links*/

	/*
	** before generating, we initial the graph to "0"
	*/
	memset(graph, 0, sizeof graph);

	/*
	** initialize 3 parts with some elements set to 1, which means adding edges to each subgraph.
	*/
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			if (randi() % 2 == 1) {
				part1[i][j] = 1; current_links += 1;
			}
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			if (randi() % 2 == 1) {
				part2[i][j] = 1; current_links += 1;
			}
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			if (randi() % 2 == 1) {
				part3[i][j] = 1; current_links += 1;
			}
		}
	}

	/*
	** if the current number of links are not equal to the total links, add or remove some links randomly.
	*/
	while (current_links != total_links) {
		int part_number = randi() % 3;
		int i = randi() % k;
		int j = randi() % k;

		switch (part_number)
		{
		case 0:
			if (current_links > total_links && part1[i][j] == 1) {
				part1[i][j] = 0; current_links -= 1;
			}
			else if (current_links < total_links && part1[i][j] == 0) {
				part1[i][j] = 1; current_links += 1;
			}
			
			break;
		case 1:
			if (current_links > total_links && part2[i][j] == 1) {
				part2[i][j] = 0; current_links -= 1;
			}
			else if (current_links < total_links && part2[i][j] == 0) {
				part2[i][j] = 1; current_links += 1;
			}

			break;
		case 2:
			if (current_links > total_links && part3[i][j] == 1) {
				part3[i][j] = 0; current_links -= 1;
			}
			else if (current_links < total_links && part3[i][j] == 0) {
				part3[i][j] = 1; current_links += 1;
			}

			break;
		default:
			printf("[PROBLEM.CPP--generate_random_graph--ERROR] no such subgraph\n");
			exit(EXIT_FAILURE);
			break;
		}
	}

	/*
	** copy each part to the graph, note that the final graph is a symmetrical matrix, which
	** means that this graph is undirected--if there is an edge between node i and node j, there
	** is also an edge between node j and node i.
	*/
	for (int i = 0; i < k; i++) {
		for (int j = k; j < 2 * k; j++) {
			graph[j][i] = graph[i][j] = part1[i][j - k];
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 2 * k; j < NODE_NUMBER; j++) {
			graph[j][i] = graph[i][j] = part2[i][j - 2 * k];
		}
	}
	for (int i = k; i < 2 * k; i++) {
		for (int j = 2 * k; j < NODE_NUMBER; j++) {
			graph[j][i] = graph[i][j] = part3[i - k][j - 2 * k];
		}
	}
}

/*save graph to a file, if the file name or extension is set to NULL, they will be set to default values*/
int save_graph(char const (*graph)[NODE_NUMBER], char const *filename, char const *extension)
{
	char file_name[200] = "";

	FILE *file = NULL;

	if (graph == NULL) {
		printf("[PROBLEM.CPP--save_graph--ERROR] you must give a correct graph array.\n");
		exit(EXIT_FAILURE);
	}

	/*
	** if file name is not given, we create a file name according to the current time.
	*/
	if (filename == NULL) {
		time_t current_time = time(NULL);
		char *p_colon = NULL;
		strcat(file_name, "graph ");
		strcat(file_name, ctime(&current_time));
		/*because time string is ended by "\n\0", we must remove the '\n'*/
		file_name[strlen(file_name) - 1] = '\0';	
		/*colon (':') will be replaced by '-', otherwise the fopen function will fail*/
		p_colon = strchr(file_name, ':');
		while (p_colon != NULL) {
			*p_colon = '-';
			p_colon = strchr(p_colon, ':');
		}
		/*
		** finally, the default file name will be something like: "graph Mon Jan 09 17-29-21 2017" 
		*/
	}
	else {
		strcat(file_name, filename);
	}

	/*
	** if file extension is not given, we set it to ".csv"
	*/
	if (extension == NULL) {
		strcat(file_name, ".csv");
	}
	else {
		strcat(file_name, extension);
	}

	/*
	** open a file stream and write graph data to it
	*/
	if ((file = fopen(file_name, "w+")) == NULL) {
		printf("[PROBLEM.CPP--save_graph--ERROR] cannot open a file.\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < NODE_NUMBER; i++) {
		for (int j = 0; j < NODE_NUMBER; j++) {
			fprintf(file, "%4d, ", graph[i][j]);
		}
		fprintf(file, "\n");
	}
	fprintf(file, "\n\n");

	fclose(file);

	return EXIT_SUCCESS;
}

/*calculate the fitness of a solution*/
double fitness(char const (*graph)[NODE_NUMBER], char const *solution)
{
	unsigned int total_links = 0;
	unsigned int conflict = 0;

	/*because graph is a symmetric array, we just use the upper triangle data*/
	for (int i = 0; i < NODE_NUMBER; i++) {
		for (int j = i; j < NODE_NUMBER; j++) {
			if (graph[i][j] == 1) {
				total_links += 1;
				if (solution[i] == solution[j]) {
					conflict += 1;
				}
			}
		}
	}

	/*the fitness value is calculated below*/
	return 1.0 - (double)conflict / total_links;
}

/*given a graph and a solution, the conflict matrix is calculated. 
conflict matrix is a matrix in which each row identifies those nodes whose colors conflict with current node.*/
void generate_conflict_matrix(char const (*graph)[NODE_NUMBER], char const *solution, char(*conflict_matrix)[NODE_NUMBER])
{
	char current_node_color;
	for (int i = 0; i < NODE_NUMBER; i++) {
		current_node_color = solution[i];
		for (int j = 0; j < NODE_NUMBER; j++) {
			if (graph[i][j] == 1 && current_node_color == solution[j]) {
				conflict_matrix[i][j] = 1;
			}
		}
	}
}

/*for each node, calculate the number of conflict to other nodes, and return the "most conflict" node*/
int solution_conflict(char const (*graph)[NODE_NUMBER], char const *solution, Conflict_Infor *conflict_infor)
{
	char current_node_color;
	int current_node_conflict = 0;
	int max_conflict = 0;
	int max_conflict_index = -1;
	int count = 0;

	/*
	** initialize  conflict information
	*/
	memset(conflict_infor->conflict_nodes, 0, NODE_NUMBER * sizeof(int));
	memset(conflict_infor->conflict_numbers, 0, NODE_NUMBER * sizeof(int));
	conflict_infor->len = 0;
	conflict_infor->max_conflict_node = -1;
	conflict_infor->max_conflict = 0;

	for (int i = 0; i < NODE_NUMBER; i++) {
		current_node_color = solution[i];
		current_node_conflict = 0;

		/*
		** calculate the conflict of current node.
		*/
		for (int j = 0; j < NODE_NUMBER; j++) {
			if (graph[i][j] == 1 && current_node_color == solution[j]) {
				current_node_conflict += 1;
			}
		}

		/*
		** if conflict > 0, save the current conflict to the conflict list
		*/
		if (current_node_conflict > 0) {
			conflict_infor->conflict_nodes[count] = i;
			conflict_infor->conflict_numbers[count] = current_node_conflict;
			count += 1;
		}

		/*
		**  update the max conflict.
		*/
		if (current_node_conflict > max_conflict) {
			max_conflict = current_node_conflict;
			max_conflict_index = i;
		}
	}

	conflict_infor->len = count;
	conflict_infor->max_conflict = max_conflict;
	conflict_infor->max_conflict_node = max_conflict_index;

	/*if return -1, it means that no conflit in this graph*/
	return max_conflict_index;
}