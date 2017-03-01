
#include "fileprocess.h"

#define COMMENT_CHAR '#'

vector< string > FileProcess::readFileList( string ipFileName){
	vector<string> opResult;
	char tmp[255];
	char str[255];
	char mFileName[255];
	strcpy ( mFileName,ipFileName.c_str());
	cout << mFileName << endl;
	FILE *fp;
	fp = fopen(mFileName,"r");
	if (fp!=NULL)
	{
		while(!feof(fp))
		{
			if (fgets(str,255,fp)==NULL){	break;	} 
			else
			{
				sscanf(str , "%s" , tmp);
				string mStr(tmp);
				opResult.push_back(mStr);
			}
		}
	}
	fclose(fp);
	memset(tmp , 0 , sizeof(tmp));
	memset(str , 0 , sizeof(str));
	memset(mFileName , 0 , sizeof(mFileName));
	return opResult;
}


bool FileProcess::IsSpace(char c)
{
	if (' ' == c || '\t' == c)
		return true;
	return false;
}

bool FileProcess::IsCommentChar(char c)
{
	switch(c) {
	case COMMENT_CHAR:
		return true;
	default:
		return false;
	}
}

void FileProcess::Trim(string & str)
{
	if (str.empty()) {
		return;
	}
	unsigned int i;
	int start_pos, end_pos;
	for (i = 0; i < str.size(); ++i) {
		if (!IsSpace(str[i])) {
			break;
		}
	}
	if (i == str.size()) { // 全部是空白字符串
		str = "";
		return;
	}

	start_pos = i;

	for (i = str.size() - 1; i >= 0; --i) {
		if (!IsSpace(str[i])) {
			break;
		}
	}
	end_pos = i;

	str = str.substr(start_pos, end_pos - start_pos + 1);
}

bool FileProcess::AnalyseLine(const string & line, string & key, string & value)
{
	if (line.empty())
		return false;
	int start_pos = 0, end_pos = line.size() - 1, pos;
	if ((pos = line.find(COMMENT_CHAR)) != -1) {
		if (0 == pos) {  // 行的第一个字符就是注释字符
			return false;
		}
		end_pos = pos - 1;
	}
	string new_line = line.substr(start_pos, start_pos + 1 - end_pos);  // 预处理，删除注释部分

	if ((pos = new_line.find('=')) == -1)
		return false;  // 没有=号

	key = new_line.substr(0, pos);
	value = new_line.substr(pos + 1, end_pos + 1- (pos + 1));

	Trim(key);
	if (key.empty()) {
		return false;
	}
	Trim(value);
	return true;
}

bool FileProcess::ReadConfig(const string & filename, map<string, string> & m)
{
	m.clear();
	cout << "read file ; " << filename << " & " << filename.c_str() << endl;
 	ifstream infile(filename.c_str());
	if (!infile) {
		cout << "file open error" << endl;
		return false;
	}
	string line, key, value;
	while (getline(infile, line)) {
		if (AnalyseLine(line, key, value)) {
			FileProcess::Trim(key);
			FileProcess::Trim(value);
			m[key] = value;
		}
	}

	infile.close();
	return true;
}

void FileProcess::PrintConfig(const map<string, string> & m)
{
	map<string, string>::const_iterator mite = m.begin();
	for (; mite != m.end(); ++mite) {
		cout << mite->first << "=" << mite->second << endl;
	}
}

