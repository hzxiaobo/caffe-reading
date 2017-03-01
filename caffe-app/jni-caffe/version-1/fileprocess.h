#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <map>
#include <cstring>

using namespace std;

////////独立的类，无依赖///////////
class FileProcess{

public:
	//////////////////////block 1////////////////
	static vector<string> readFileList(string ipFileName);
	//////////////////////block 2///////////////
	static bool ReadConfig(const string & filename, map<string, string> & m);
	static void PrintConfig(const map<string, string> & m);
	/////////////////////block 3///////////////


private:
	//////////////////////block 1//////////////////
	//////////////////////block 2/////////////////
	static bool IsSpace(char c);
	static bool IsCommentChar(char c);
	static void Trim(string & str);
	static bool AnalyseLine(const string & line, string & key, string & value);
	///////////////////////block 3////////////////
};
