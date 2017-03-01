#include <jni.h>
#include <string>
#include "classjni.h"
#include "caffeclassifier.h"
#include "detector.h"

map<int, Classifier*> modelPool;
map<int, Detector*> detectorPool;
/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    initModel
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IIIFFF)V
 */
JNIEXPORT void JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_initModel
(
    JNIEnv *jenv, 
    jclass jcls, 
    jstring jnetPath, 
    jstring jweightsPath, 
    jstring jlabelPath, 
    jint deviceId, 
    jint threadNum,
    jint squareSideLength, 
    jfloat blueOfMean, 
    jfloat greenOfMean, 
    jfloat redOfMean
) 
{
    try {
        const char* netPathPtr = jenv->GetStringUTFChars(jnetPath, NULL);
        std::string netPath(netPathPtr);
        const char* weightsPathPtr = jenv->GetStringUTFChars(jweightsPath, NULL);
        std::string weightsPath(weightsPathPtr);
        const char* labelPathPtr = jenv->GetStringUTFChars(jlabelPath, NULL);
        std::string labelPath(labelPathPtr);
        for (int i = 0; i < threadNum; ++i ) {
            Classifier* classifier = new Classifier(deviceId, 
                                                    netPath, 
                                                    weightsPath,
                                                    labelPath, 
                                                    squareSideLength, 
                                                    blueOfMean, 
                                                    greenOfMean, 
                                                    redOfMean);
            classifier->setResizeSize(squareSideLength);
			classifier->setClassId(i);
            modelPool.insert(pair<int, Classifier*>(i, classifier));
        }

        std::cout << "net path is: " << netPath << std::endl;


        // realease resources;
        if (netPathPtr) {
            jenv->ReleaseStringUTFChars(jnetPath, netPathPtr);
        }
        if (weightsPathPtr) {
            jenv->ReleaseStringUTFChars(jweightsPath, weightsPathPtr);
        }
        if (labelPathPtr) {
            jenv->ReleaseStringUTFChars(jlabelPath, labelPathPtr);
        }
        
    } catch (Exception e) {
        
    }
}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    extractOutputOfLayer
 * Signature: ([BLjava/lang/String;I)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_extractOutputOfLayer
(
    JNIEnv *jenv, 
    jclass jcls, 
    jbyteArray jimageBytes, 
    jstring jlayerName, 
    jint threadId
)
{
    jfloatArray joutputArray = NULL;
    try {        
        if (modelPool.count(threadId)) {
            const char* layerNamePtr = jenv->GetStringUTFChars(jlayerName, NULL);        
            Classifier* classifier = modelPool[threadId];
            jsize len = jenv->GetArrayLength(jimageBytes);
            jbyte* bytePtr = jenv->GetByteArrayElements(jimageBytes, 0);
			vector<char> imgVec(len); //have been clear
			// cout << "the length of image is: " << imgVec.size() << " classifier is: " << threadId << endl;
			char* mpDst = &imgVec[0]; //已指向null
			memcpy(mpDst, bytePtr, len);
			Mat img = imdecode(imgVec, -1);
            
            shared_ptr<Blob<float> > outputBlob = classifier->GetLayerOutput(img, string(layerNamePtr));
            
            joutputArray = jenv->NewFloatArray(outputBlob->count());            
            if (joutputArray != NULL) {
                jenv->SetFloatArrayRegion(joutputArray, 0, outputBlob->count(), outputBlob->cpu_data());
            }      

            // realease resources
            imgVec.clear();
            mpDst = NULL;
            if (layerNamePtr != NULL) {
                jenv->ReleaseStringUTFChars(jlayerName, layerNamePtr);
            }
            if (bytePtr != NULL) {
                jenv->ReleaseByteArrayElements(jimageBytes, bytePtr, 0);
            }    
                            
        }
    } catch (Exception e) {

    }
    return joutputArray;
}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    forwardNet
 * Signature: ([BI)V
 */
JNIEXPORT void JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_forwardNet
(
    JNIEnv *jenv,
    jclass jcls,
    jbyteArray jimageBytes,
    jint threadId)
{
    try {
        if (modelPool.count(threadId)) {
            Classifier* classifier = modelPool[threadId];
            jsize len = jenv->GetArrayLength(jimageBytes);
            jbyte* bytePtr = jenv->GetByteArrayElements(jimageBytes, 0);
            vector<char> imgVec(len); //have been clear
            // cout << "the length of image is: " << imgVec.size() << " classifier is: " << threadId << endl;
            char* mpDst = &imgVec[0]; //已指向null
            memcpy(mpDst, bytePtr, len);
            Mat img = imdecode(imgVec, -1);

            classifier->Forward(img);

            // realease resources
            imgVec.clear();
            mpDst = NULL;

            if (bytePtr != NULL) {
                jenv->ReleaseByteArrayElements(jimageBytes, bytePtr, 0);
            }

        }
    } catch (Exception e) {

    }
}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    getBlobOutputByName
 * Signature: (Ljava/lang/String;I)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_getBlobOutputByName
(
    JNIEnv *jenv,
    jclass jcls,
    jstring jblobName,
    jint threadId)
{
    jfloatArray joutputArray = NULL;
    try {
        if (modelPool.count(threadId)) {
            const char* blobNamePtr = jenv->GetStringUTFChars(jblobName, NULL);
            Classifier* classifier = modelPool[threadId];

            shared_ptr<Blob<float> > outputBlob = classifier->GetBlobByName(string(blobNamePtr));

            joutputArray = jenv->NewFloatArray(outputBlob->count());
            if (joutputArray != NULL) {
                jenv->SetFloatArrayRegion(joutputArray, 0, outputBlob->count(), outputBlob->cpu_data());
            }

            // realease resources
            if (blobNamePtr != NULL) {
                jenv->ReleaseStringUTFChars(jblobName, blobNamePtr);
            }
        }
    } catch (Exception e) {

    }
    return joutputArray;
}


/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    targetCheck
 * Signature: ([BI)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_targetCheck
(
    JNIEnv *jenv, 
    jclass jcls, 
    jbyteArray jimageBytes, 
    jint threadId)
{
    jstring joutput = 0;
    const char* outputStrPtr = NULL;
    try {
        if (modelPool.count(threadId)) {
            Classifier* classifier = modelPool[threadId];
            jsize len = jenv->GetArrayLength(jimageBytes);
            jbyte* bytePtr = jenv->GetByteArrayElements(jimageBytes, 0);

			vector<char> imgVec(len); //have been clear	
			char* mpDst = &imgVec[0]; //已指向null
			memcpy(mpDst, bytePtr, len);
			Mat img = imdecode(imgVec, -1);

            std::string outputString = classifier->CheckTarget(img);

            outputStrPtr = outputString.c_str();
            if (outputStrPtr != NULL) {
                joutput = jenv->NewStringUTF(outputStrPtr);
            }
            // release resources
            if (bytePtr != NULL) {
                jenv->ReleaseByteArrayElements(jimageBytes, bytePtr, 0);
            }  
            imgVec.clear();
            mpDst = NULL;  
        }
    } catch (Exception e) {

    }
    return joutput;
}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    initDetector
 * Signature: (Ljava/lang/String;Ljava/lang/String;I)V
 */
JNIEXPORT void JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_initDetector
(
    JNIEnv* jenv, 
    jclass jcls, 
    jstring jfrontalPath, 
    jstring jprofilePath, 
    jint threadNum)
{
    try {
        const char* frontalPathPtr = jenv->GetStringUTFChars(jfrontalPath, NULL);
        const char* profilePathPtr = jenv->GetStringUTFChars(jprofilePath, NULL);
        string frontalPath(frontalPathPtr);
        string profilePath(profilePathPtr);
        for (int i = 0; i < threadNum; ++i) {
            Detector* detector = new Detector(frontalPath, profilePath);
            detectorPool.insert(pair<int, Detector*>(i, detector));   
        }

        if (frontalPathPtr) {
            jenv->ReleaseStringUTFChars(jfrontalPath, frontalPathPtr);
        }
        if (profilePathPtr) {
            jenv->ReleaseStringUTFChars(jprofilePath, profilePathPtr);
        }
    } catch (Exception e) {

    }

}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    detectAndVerifyFace
 * Signature: ([BI)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_detectAndVerifyFace
(
    JNIEnv *jenv, 
    jclass jcls, 
    jbyteArray jimageBytes, 
    jint threadId
)
{
    jfloatArray joutputArray = NULL;
    try {
        if (detectorPool.count(threadId)) {
            jsize len = jenv->GetArrayLength(jimageBytes);
            jbyte* bytePtr = jenv->GetByteArrayElements(jimageBytes, 0);
            vector<char> imgVec(len); //have been clear	
			char* mpDst = &imgVec[0]; //已指向null
			memcpy(mpDst, bytePtr, len);
            Mat img = imdecode(imgVec, -1);
            
            Detector * detector = detectorPool[threadId];
            vector<Rect> faces = detector->detect(img);
            std::cout << "face: " << faces.size() << std::endl;

            if (faces.size() > 0) {
                vector<float> faces_feat;
                Classifier* classifier = modelPool[threadId];
                for ( int i = 0; i < faces.size(); ++i ) {
                    Mat face(img, faces[i]);
                    Mat crop;
                    resize(face, crop, Size(224, 224));
                    shared_ptr<Blob<float> > outputBlob = classifier->GetLayerOutput(crop, "fc7");
                    const float* begin = outputBlob->cpu_data();
                    faces_feat.insert(faces_feat.end(), begin, begin+outputBlob->count());
                }
                joutputArray = jenv->NewFloatArray(faces_feat.size());
                if (joutputArray != NULL) {
                    jenv->SetFloatArrayRegion(joutputArray, 0, faces_feat.size(), &faces_feat[0]);
                } 
            }
            
            // release resources
            if (bytePtr != NULL) {
                jenv->ReleaseByteArrayElements(jimageBytes, bytePtr, 0);
            }  
            imgVec.clear();
            mpDst = NULL;  
        }
    } catch (Exception e) {

    }
    return joutputArray;
}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    verifyFace
 * Signature: ([BI)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_verifyFace
(
    JNIEnv *jenv, 
    jclass jcls, 
    jbyteArray jimageBytes, 
    jint threadId
) 
{
    jstring joutput = 0;
    const char* outputStrPtr = NULL;
    try {
        if (detectorPool.count(threadId)) {
            jsize len = jenv->GetArrayLength(jimageBytes);
            jbyte* bytePtr = jenv->GetByteArrayElements(jimageBytes, 0);
            vector<char> imgVec(len); //have been clear	
			char* mpDst = &imgVec[0]; //已指向null
			memcpy(mpDst, bytePtr, len);
            Mat img = imdecode(imgVec, -1);
            
            Detector * detector = detectorPool[threadId];
            vector<Rect> faces = detector->detect(img);
            std::cout << "face: " << faces.size() << std::endl;

            if (faces.size() > 0) {
                string final_str = "";
                // vector<float> faces_feat;
                Classifier* classifier = modelPool[threadId];
                for ( int i = 0; i < faces.size(); ++i ) {
                    Mat face(img, faces[i]);
                    Mat crop;
                    resize(face, crop, Size(224, 224));
                    string result_str = classifier->CheckTarget(crop); 
                    final_str += result_str;
                    if (i != faces.size() - 1) {
                        final_str += "|";
                    }
                }
                // cout << "feat size: " << faces_feat.size() << endl;
                outputStrPtr = final_str.c_str();
                if (outputStrPtr != NULL) {
                    joutput = jenv->NewStringUTF(outputStrPtr);
                }    
            }  
            // realease resources            
            if (bytePtr != NULL) {
                jenv->ReleaseByteArrayElements(jimageBytes, bytePtr, 0);
            }  
            faces.clear();
            imgVec.clear();
            mpDst = NULL;                
        }
    } catch (Exception e) {
        std::cout << "verify face exceptions" << std::endl;
    }
    return joutput;
}