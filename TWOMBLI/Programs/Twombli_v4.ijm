/*
	29-10-20
	Written by Esther Wershof and David Barry
*/

/*
 INSTRUCTIONS:
 - It's a good idea to read the documentation and try the tutorial first to get a feel for what it does
 - If you want to start a new .ijm script (eg for testing a specific function) Click File->New, then Language->IJ1 Macro
 - All built in macro functions can be found here: https://imagej.nih.gov/ij/developer/macro/functions.html
 - setBatchMode(true) is used to suppress the output ie if you do something to many files, don't want to open them all.
 - The code for computing HDM is called QBS = QuantBlackSpace. The image is processed so that most pixels are black (=0) with any nonzero pixels
 	corresponding to areas of high-density matrix. The output is a separate _resultsHDM.csv file which gives the proportion of black pixels.
 	HDM is then equal to 1-numberBlackPixels.
 - anamorfProperties.xml is a .xml file that input additional parameters for anamorf to run
 - Functions are all at the bottom of the script
 - Functions to be performed in batch mode ie on all files in a folder have this structure:
 
 function doSomethingToFolder(input,output) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i])) // ie if you have a folder within a folder, it goes down another level
{
			doSomethingToFolder(input + File.separator + list[i]);
		} else {
			doSomethingToFILE(input, output, list[i]);
		}
	}
}

 */

/*
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
	0. BEFORE IT GETS STARTED...
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
*/
run("Options...", "iterations=1 count=1");

// closes and clears anything already open
run("Close All");

if (isOpen("Log")) { 
 selectWindow("Log"); 
 run("Close"); 
} 

// declaring global variables
var DARK_LINE = "Dark Line";
var CONTRAST_SATURATION = "Contrast Saturation";
var MIN_LINE_WIDTH = "Min Line Width";
var MAX_LINE_WIDTH = "Max Line Width";
var LINE_WIDTH_STEP = "Line Width Step";
var LOW_CONTRAST = "Low Contrast";
var HIGH_CONTRAST = "High Contrast";
var MIN_LINE_LENGTH = "Min Line Length";
var MIN_CURVATURE_WINDOW = "Min Curvature Window";
var MAX_CURVATURE_WINDOW = "Max Curvature Window";
var MINIMUM_BRANCH_LENGTH = "Minimum Branch Length";
var MAXIMUM_DISPLAY_HDM = "Maximum Display HDM";
//var GAP_ANALYSIS = "GapAnalysis";
var MINIMUM_GAP_DIAMETER = "Minimum Gap Diameter";
var DELIM = ",";
var EOL = "\n";
var PARAM_FILE_NAME = "parameters.txt";
var HDM_RESULTS_FILENAME = "_ResultsHDM.csv";
var ANAMORF_RESULTS_FILENAME = "results.csv";
var TWOMBLI_RESULTS_FILENAME = "Twombli_Results";

// global variables with default values
var contrastSaturation = 0.35;
var minLineWidth = 5;
var maxLineWidth = 5;
var lineWidthStep = 1;
var minCurvatureWindow = 40;
var maxCurvatureWindow = 40;
var curvatureWindowStep = 10;
var minimumBranchLength = 10;
var maximumDisplayHDM = 200;
var gapAnalysis = false;
var minimumGapDiameter = 0;

var minLineWidthTest = 5;
var maxLineWidthTest = 20;

var contrastHigh = 120.0;
var contrastLow = 0.0;
var minLineLength = 10.0;
var outputMasks = "";
var darkLineValue = 1;

var darkline;

/*
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
	1. PRE CHECKS
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
*/
setLogWindow(1.0, 25); // maximize log window

// print("Maximise the log then click OK"); // ask user to maximimse log
// waitForUser("Maximise the log then click OK");
// if (isOpen("Log")) { 				// reposition in top left corner
//      selectWindow("Log"); 
//      setLocation(0, 0); 
//   } 

print("\nPrechecks...");
args = getArgument();
args_list = split(args, ",");

parameterFile = args_list[0];
img_dir = args_list[1];
mask_dir = args_list[2];
hdm_dir = args_list[3];
road_dir = args_list[4];
anamorf_param_path = args_list[5];

loadParameterFile(parameterFile);
if(minimumGapDiameter != 0){
	gapAnalysis=true;
}

if (darkLineValue != 0)
{
	darkline = true;
	print("The program assumes that matrix fibres are dark on a light background.");
} else {
	darkline = false;
	print("The program assumes that matrix fibres are light on a dark background.");
}
File.saveString(getInfo("log"), File.getParent(img_dir)+"/FIJI_log.txt");

// print("The program assumes that matrix fibres are dark on a light background.");
// darkline = true;

/*
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
	3. RIDGE DETECTION
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 -----------------------------------------------------------------------------------------------------------------------------
 */
 
// computes ridge detection (to make masks) on all files in eligible folder
T1 = getTime();
print("\nStep 1: Choosing directories and files");
wait(11);

inputEligible = img_dir;
outputMasks = mask_dir;
outputHDM = hdm_dir;
outputRoads = road_dir;
anamorfProperties = anamorf_param_path;


print("\nGood, you have chosen all necessary files. Twombli is ready to process all images");
wait(11);
File.saveString(getInfo("log"), File.getParent(inputEligible)+"/FIJI_log.txt");

print("\nStep 2: ridge detection");
wait(11);


IJ.redirectErrorMessages();

print("Detecting ridges");
processFolderRidgeDetection(inputEligible, outputMasks, false);

print("Estimating widths...");
processFolderRidgeDetection(inputEligible, outputRoads, true);

print("FINISHED deriving masks!");
close("*");

T2 = getTime();
print("Ridge detection took " + (T2-T1)/1000 + " seconds.");

File.saveString(getInfo("log"), File.getParent(inputEligible)+"/FIJI_log.txt");

/*
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
	4. ANAMORF
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
*/

// computes Anamorf (where most of the metrics are derived) on all eligible files
T1 = getTime();
inputAnamorf = outputMasks;
wait(100);
print("\nStep 3: extracting parameters with Anamorf");


run("AnaMorf Macro Extensions");
Ext.initialiseAnaMorf(inputAnamorf, anamorfProperties);
// anamorfProperties is a .xml file - for now just leave this. Might be able to remove in future

Ext.setAnaMorfFileType("PNG");
// don't worry about this - the masks generated are of png type, so this should be left alone
print("Analyzing for curvature window: " + minCurvatureWindow);
Ext.setAnaMorfCurvatureWindow(minCurvatureWindow);
Ext.setAnaMorfMinBranchLength(minimumBranchLength);
Ext.runAnaMorf();

for(c = minCurvatureWindow + curvatureWindowStep; c <= maxCurvatureWindow; c += curvatureWindowStep){
    print("Analyzing for curvature window: " + c);
	Ext.resetParameters();
	Ext.setAnaMorfCurvatureWindow(c);
	Ext.runAnaMorf();	
}

wait(11);
print("HERE exiting anamorf");
File.saveString(getInfo("log"), File.getParent(inputEligible)+"/FIJI_log.txt");

T2 = getTime();
print("Anamorf took " + (T2-T1)/1000 + " seconds.");

wait(11);
T1 = getTime();
print("Now computing alignment");
alignmentVec = newArray(0);
alignmentVec = processFolderAlignment(outputMasks);	
var dimensionsVec = processFolderDimensions(outputMasks);
var alignmentVecOrder = getAlignmentVecOrder(outputMasks);
print("Finished computing alignment");
close("*");

T2 = getTime();
print("Alignment calculation took " + (T2-T1)/1000 + " seconds.");
File.saveString(getInfo("log"), File.getParent(inputEligible)+"/FIJI_log.txt");

/*
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
	4. HDM
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
*/
// Computes HDM on all Eligible images
wait(11);
T1 = getTime();
print("\nStep 4: Computing HDM");
run("Clear Results");
inputHDM = inputEligible;

IJ.redirectErrorMessages();
processFolderHDM(inputHDM, outputHDM);

// The code for computing HDM is called QBS = QuantBlackSpace
setBatchMode(true);
pathToQBS = searchForFile(getDirectory("imagej"), "", "Quant_Black_Space.ijm");
runMacro(pathToQBS, outputHDM);
saveAs("Results", outputHDM + File.separator + HDM_RESULTS_FILENAME);
close("Results");

setBatchMode(false);
print("FINISHED HDM!");

T2 = getTime();
print("Computing HDM took " + (T2-T1)/1000 + " seconds.");
File.saveString(getInfo("log"), File.getParent(inputEligible)+"/FIJI_log.txt");

/*
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
	5. Gap analysis
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
*/

// print("gap analysis = ", gapAnalysis);
gapAnalysis = false;
if (gapAnalysis == true)
{
	wait(11);
	print("\nOptional Step 5: Computing Gap analysis");
	// print("minimumGapDiameter = ", minimumGapDiameter);

	File.makeDirectory(outputMasks+"/GapAnalysis");
	T1 = getTime();
	processFolderGap(outputMasks);
	closeROI();
	
	if (isOpen("Results")) {
		selectWindow("Results");
		run("Close");
	}
	
	print("FINISHED GAP ANALYSIS!");
	T2 = getTime();
	print(" Gap analysis took " + (T2-T1)/1000 + " seconds.");
	wait(11);
	print("Gap analysis can be found in output masks folder");
	print("containing masks with gaps shown, and a summary txt file");
}
print("Just before tidyResults");
tidyResults(outputHDM, inputAnamorf, inputEligible, alignmentVec);
print("FINISHED EVERYTHING!");
wait(11);
File.saveString(getInfo("log"), File.getParent(inputEligible)+"/FIJI_log.txt");
run("Close All");

if (isOpen("Log")) { 
 selectWindow("Log"); 
 run("Close"); 
} 
/*
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
	5. LIST OF FUNCTIONS
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------------------
*/
/*
FUNCTION OVERVIEW (indentations mean functions inside other functions)

openFolder(input)
	openFile (input, file)
processFolderRidgeDetection(input, output)
	processFileRidgeDetection (input, output, file)
		runMultiScaleRidgeDetection()
			printResult(lineWidth)
		calcSigma()
		calcLowerThresh(estimatedSigma)
		calcUpperThresh(estimatedSigma)
countFiles(dir)
processFolderHDM(inputHDM, outputHDM)
	processFileHDM(inputHDM, outputHDM, file)
searchForFile(input, targetPath, target)
saveParameterFile()
loadParameterFile()
processFolderGap(input)
	processFileGap(input)
		append(arr, value) 
		percentile(arr,p)
		closeROI()
	processFolderAlignment()
		processFileAlignment()
tidyResults()
*/

// function to open all files in a folder
function openFolder(input) { 
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
{
			openTestSetFolder(input + File.separator + list[i]);
		} else {
			openFile(input, list[i]);
		}
	}
}


function openFolderSpecific(input) { 
	list = getFileList(input);
	list = Array.sort(list);
	suffix1 = "LENGTH_" + minimumBranchLength + ".png";
		for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
		{
			openTestSetFolder(input + File.separator + list[i]);
		} else {
			if(endsWith(list[i], suffix1)){
				openFile(input, list[i]);
			}
			}
		}
	}


function openFile(input, file) {
	print("opening file " + input + File.separator + file);
	open(input + File.separator + file);
	run("Out [-]"); // zoom out
}	

	
// function to scan folders/subfolders/files to find files with correct suffix
function processFolderRidgeDetection(input, output, with_width) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i])) {
			processFolderRidgeDetection(input + File.separator + list[i], output, with_width);
		} else {
			processFileRidgeDetection(input, output, list[i], with_width);
		}
	}
}

function processFileRidgeDetection(input, output, file, with_width) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	setBatchMode(true);
	open(input + File.separator + file);
	run("Out [-]");
	print("Processing: " + input + File.separator + file);
	
	if (contrastSaturation > 0) {
	    print("Applying contrast enhancement with saturation=" + contrastSaturation);
		run("Enhance Contrast", "saturated=" + contrastSaturation);
	}
    
    print("Performing weighted RGB to gray");
	run("Conversions...", "scale weighted");
	run("8-bit");

	runMultiScaleRidgeDetection(with_width);

	// Invert LUT
	getLut(reds, greens, blues);
	for (i=0; i<reds.length; i++) {
	    reds[i] = 255-reds[i];
	    greens[i] = 255-greens[i];
	    blues[i] = 255-blues[i];
	}
	
	setLut(reds, greens, blues);
	run("Invert");

	saveAs("PNG", output + File.separator + file);
	
	IJ.redirectErrorMessages() ;
	titles = getList("image.titles");


	for(j=0; j<titles.length;j++){
		print("titles[" + j + "] = ", titles[j]);
		selectWindow(titles[j]);
		close();
	}
	
	print("Saving to: " + output);
	setBatchMode(false);
}

function runMultiScaleRidgeDetection(width_width){
	print("Running Ridge detection");
	input = getTitle();
	sigma = calcSigma(minLineWidth);
	lowerThresh = calcLowerThresh(minLineWidth, sigma);
	upperThresh = calcUpperThresh(minLineWidth, sigma);
	// print("minLineWidth = ", minLineWidth);
	// print("maxLineWidth = ", maxLineWidth);
	// print("lineWidthStep = ", lineWidthStep);
	// print("sigma = ", sigma);
    // print("lowerThresh = ", lowerThresh);
    // print("upperThresh = ", upperThresh);
    // print("minimum_line_length = ", minLineLength);
	print("Line Width = ", minLineWidth);
    if (with_width == true) {
        if(darkline){
			run("Ridge Detection", "line_width=" + minLineWidth + " high_contrast=" + contrastHigh + " low_contrast=" + contrastLow + " darkline correct_position estimate_width extend_line make_binary method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThresh + " upper_threshold=" + upperThresh + " minimum_line_length=" + minLineLength + " maximum=0");
		} else{
			run("Ridge Detection", "line_width=" + minLineWidth + " high_contrast=" + contrastHigh + " low_contrast=" + contrastLow + " correct_position estimate_width extend_line make_binary method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThresh + " upper_threshold=" + upperThresh + " minimum_line_length=" + minLineLength + " maximum=0");
		}
		result = getTitle();
		
		lw = minLineWidth + lineWidthStep;
		while(lw <= maxLineWidth){
		    print("Line Width = ", lw);
			sigma = calcSigma(lw);
			lowerThresh = calcLowerThresh(lw, sigma);
			upperThresh = calcUpperThresh(lw, sigma);
			// print("sigma = ", sigma);
	        // print("lowerThresh = ", lowerThresh);
	        // print("upperThresh = ", upperThresh);
	        // print("minLineLength = ", minLineLength);
			selectWindow(input);
			if(darkline){
				run("Ridge Detection", "line_width=" + lw + " high_contrast=" + contrastHigh + " low_contrast=" + contrastLow + " darkline correct_position estimate_width extend_line make_binary method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThresh + " upper_threshold=" + upperThresh + " minimum_line_length=" + minLineLength + " maximum=0");
			} else{
				run("Ridge Detection", "line_width=" + lw + " high_contrast=" + contrastHigh + " low_contrast=" + contrastLow + " correct_position estimate_width extend_line make_binary method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThresh + " upper_threshold=" + upperThresh + " minimum_line_length=" + minLineLength + " maximum=0");
			}
			this_result = getTitle();
			imageCalculator("OR create", result, this_result);
			temp = result;
			result = getTitle();
			close(temp);
			close(this_result);
			lw = lw + lineWidthStep;
		}

    } else {
		if(darkline){
			run("Ridge Detection", "line_width=" + minLineWidth + " high_contrast=" + contrastHigh + " low_contrast=" + contrastLow + " darkline correct_position extend_line make_binary method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThresh + " upper_threshold=" + upperThresh + " minimum_line_length=" + minLineLength + " maximum=0");
		} else{
			run("Ridge Detection", "line_width=" + minLineWidth + " high_contrast=" + contrastHigh + " low_contrast=" + contrastLow + " correct_position extend_line make_binary method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThresh + " upper_threshold=" + upperThresh + " minimum_line_length=" + minLineLength + " maximum=0");
		}
		result = getTitle();
		
		lw = minLineWidth + lineWidthStep;
		while(lw <= maxLineWidth){
		    print("Line Width = ", lw);
			sigma = calcSigma(lw);
			lowerThresh = calcLowerThresh(lw, sigma);
			upperThresh = calcUpperThresh(lw, sigma);
			// print("sigma = ", sigma);
	        // print("lowerThresh = ", lowerThresh);
	        // print("upperThresh = ", upperThresh);
	        // print("minLineLength = ", minLineLength);
			selectWindow(input);
			if(darkline){
				run("Ridge Detection", "line_width=" + lw + " high_contrast=" + contrastHigh + " low_contrast=" + contrastLow + " darkline correct_position extend_line make_binary method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThresh + " upper_threshold=" + upperThresh + " minimum_line_length=" + minLineLength + " maximum=0");
			} else{
				run("Ridge Detection", "line_width=" + lw + " high_contrast=" + contrastHigh + " low_contrast=" + contrastLow + " correct_position extend_line make_binary method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThresh + " upper_threshold=" + upperThresh + " minimum_line_length=" + minLineLength + " maximum=0");
			}
			this_result = getTitle();
			imageCalculator("OR create", result, this_result);
			temp = result;
			result = getTitle();
			close(temp);
			close(this_result);
			lw = lw + lineWidthStep;
		}
	}
}

function printResult(lineWidth){
	getHistogram(values, counts, 256);
	setResult("Line Width " + lineWidth, nResults-1, counts[255]);
}

// called before processFolderHDM - if folder is not empty, macro might not work.
function countFiles(dir) {
      count=0;
      list = getFileList(dir);
      for (i=0; i<list.length; i++) {
          if (endsWith(list[i], "/"))
              countFiles(""+dir+list[i]);
          else
              count++;
      }
      if(count!=0)
      {
			print("\nWarning! The output HDM folder you have chosen is not empty so macro will not run. Empty this folder before proceeding.");
			waitForUser("When you have emptied the output HDM folder, click OK to proceed");
			
      }
  }

// function to scan folders/subfolders/files to find files with correct suffix
function processFolderHDM(inputHDM,outputHDM) {
	list = getFileList(inputHDM);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(inputHDM + File.separator + list[i]))
			processFolderHDM(inputHDM + File.separator + list[i]);
			processFileHDM(inputHDM, outputHDM, list[i]);
	}
}

function processFileHDM(inputHDM, outputHDM, file) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.

	setBatchMode(true);
	open(inputHDM + File.separator + file);
	run("Out [-]");
	run("8-bit");
	if(darkline==false){
		run("Invert");
	}
	setMinAndMax(0, maximumDisplayHDM);
	run("Apply LUT");
	run("Invert");
	run("Enhance Contrast", "saturated=" + contrastSaturation);
	saveAs("PNG", outputHDM + File.separator + file);
	titles = getList("image.titles");

	for(j=0; j<titles.length;j++){
		print("titles[" + j + "] = ", titles[j]);
		selectWindow(titles[j]);
		close();
	}
	setBatchMode(false);
}

function searchForFile(input, targetPath, target) {
      list = getFileList(input);
      list = Array.sort(list);
      for (i = 0; i < list.length; i++) {
            if(File.isDirectory(input + File.separator + list[i]))
                  targetPath = searchForFile(input + File.separator + list[i], targetPath, target);
            if(startsWith(list[i], target)){
                  return input + File.separator + list[i];
            }
      }
      return targetPath;
}

function saveParameterFile(){
	print("Save the parameter file somewhere sensible so you can use these parameters again");
	parameterFilePath = getDirectory("Specify location for parameter file") + File.separator + PARAM_FILE_NAME;
	print("Saving parameter file in " + parameterFilePath);
	if(File.exists(parameterFilePath)){
		File.delete(parameterFilePath);
	}
	parameterFile = File.open(parameterFilePath);

	print(parameterFile, CONTRAST_SATURATION + DELIM + contrastSaturation + EOL);
	print(parameterFile, MIN_LINE_WIDTH + DELIM + minLineWidth + EOL);
	print(parameterFile, MAX_LINE_WIDTH + DELIM + maxLineWidth + EOL);
	print(parameterFile, LINE_WIDTH_STEP + DELIM + lineWidthStep + EOL);
	print(parameterFile, MIN_CURVATURE_WINDOW + DELIM + minCurvatureWindow + EOL);
	print(parameterFile, MAX_CURVATURE_WINDOW + DELIM + maxCurvatureWindow + EOL);
	print(parameterFile, MINIMUM_BRANCH_LENGTH + DELIM + minimumBranchLength + EOL);
	print(parameterFile, MAXIMUM_DISPLAY_HDM + DELIM + maximumDisplayHDM + EOL);
	print(parameterFile, MINIMUM_GAP_DIAMETER + DELIM + minimumGapDiameter + EOL);

	File.close(parameterFile);
}


function loadParameterFile(){
	parameterFile = File.openDialog("Specify location of parameter file");
	parameterString = File.openAsString(parameterFile);

	lines = split(parameterString, EOL);

	nLines = lengthOf(lines);

	for (i = 0; i < nLines; i++) {
		words = split(lines[i], DELIM);
		if(startsWith(words[0], CONTRAST_SATURATION)){
			contrastSaturation = parseFloat(words[1]);
		} else if(startsWith(words[0], MIN_LINE_WIDTH)){
			minLineWidth = parseFloat(words[1]);
		} else if(startsWith(words[0], MAX_LINE_WIDTH)){
			maxLineWidth = parseFloat(words[1]);
		} else if(startsWith(words[0], LINE_WIDTH_STEP)){
		    lineWidthStep = parseFloat(words[1]);
		} else if(startsWith(words[0], LOW_CONTRAST)){
		     contrastLow = parseFloat(words[1]);
		} else if(startsWith(words[0], HIGH_CONTRAST)){
		     contrastHigh = parseFloat(words[1]);
		} else if(startsWith(words[0], MIN_LINE_LENGTH)){
		     minLineLength = parseFloat(words[1]);
		} else if(startsWith(words[0], MIN_CURVATURE_WINDOW)){
			minCurvatureWindow = parseFloat(words[1]);
		} else if(startsWith(words[0], MAX_CURVATURE_WINDOW)){
			maxCurvatureWindow = parseFloat(words[1]);
		} else if(startsWith(words[0], MINIMUM_BRANCH_LENGTH)){
			minimumBranchLength = parseFloat(words[1]);
		} else if(startsWith(words[0], MAXIMUM_DISPLAY_HDM)){
			maximumDisplayHDM = parseFloat(words[1]);
		} else if(startsWith(words[0], MINIMUM_GAP_DIAMETER)){
			minimumGapDiameter = parseFloat(words[1]);
		}
	}
}


function loadParameterFile(parameterFile){
	print("Loading parameters from " + parameterFile);
	parameterString = File.openAsString(parameterFile);

	lines = split(parameterString, EOL);

	nLines = lengthOf(lines);

	for (i = 0; i < nLines; i++) {
		words = split(lines[i], DELIM);
		if(startsWith(words[0], DARK_LINE)) {
		    darkLineValue = parseFloat(words[1]);
		} else if(startsWith(words[0], CONTRAST_SATURATION)){
			contrastSaturation = parseFloat(words[1]);
		} else if(startsWith(words[0], MIN_LINE_WIDTH)){
			minLineWidth = parseFloat(words[1]);
		} else if(startsWith(words[0], MAX_LINE_WIDTH)){
			maxLineWidth = parseFloat(words[1]);
		} else if(startsWith(words[0], LINE_WIDTH_STEP)){
		    lineWidthStep = parseFloat(words[1]);
		} else if(startsWith(words[0], LOW_CONTRAST)){
		     contrastLow = parseFloat(words[1]);
		} else if(startsWith(words[0], HIGH_CONTRAST)){
		     contrastHigh = parseFloat(words[1]);
		} else if(startsWith(words[0], MIN_LINE_LENGTH)){
		     minLineLength = parseFloat(words[1]);
		} else if(startsWith(words[0], MIN_CURVATURE_WINDOW)){
			minCurvatureWindow = parseFloat(words[1]);
		} else if(startsWith(words[0], MAX_CURVATURE_WINDOW)){
			maxCurvatureWindow = parseFloat(words[1]);
		} else if(startsWith(words[0], MINIMUM_BRANCH_LENGTH)){
			minimumBranchLength = parseFloat(words[1]);
		} else if(startsWith(words[0], MAXIMUM_DISPLAY_HDM)){
			maximumDisplayHDM = parseFloat(words[1]);
		} else if(startsWith(words[0], MINIMUM_GAP_DIAMETER)){
			minimumGapDiameter = parseFloat(words[1]);
		}
	}
}


// These functions are used to calculate thresholds in ridge detection
// Functions to calculate ridge detection parameters, taken from:
// https://github.com/thorstenwagner/ij-ridgedetection/blob/master/src/main/java/de/biomedical_imaging/ij/steger/Lines_.java

function calcSigma(lineWidth){
	return lineWidth / (2 * sqrt(3)) + 0.5;
}

function calcLowerThresh(lineWidth, estimatedSigma){
	clow=contrastLow;
	if(darkline){
		clow = 255 - contrastHigh;
	}
	return 0.17 * floor(abs(-2 * clow * (lineWidth / 2.0)
					/ (sqrt(2 * PI) * estimatedSigma * estimatedSigma * estimatedSigma)
					* exp(-((lineWidth / 2.0) * (lineWidth / 2.0)) / (2 * estimatedSigma * estimatedSigma))));
}

function calcUpperThresh(lineWidth, estimatedSigma){
	chigh = contrastHigh;
	if (darkline) {
		chigh = 255 - contrastLow;
	}
	return 0.17 * floor(abs(-2 * chigh * (lineWidth / 2.0)
					/ (sqrt(2 * PI) * estimatedSigma * estimatedSigma * estimatedSigma)
					* exp(-((lineWidth / 2.0) * (lineWidth / 2.0)) / (2 * estimatedSigma * estimatedSigma))));
}

function processFolderGap(input) {
	list = getFileList(input);
	list = Array.sort(list);
	gapAnalysisFile = File.open(input + "/GapAnalysis/GapAnalysisSummary.txt");
	print(gapAnalysisFile,  "filename mean sd percentile5 median percentile95");
	for (i = 0; i < list.length; i++) {
		if(endsWith(list[i],"png"))
		{
			processFileGap(input, list[i],gapAnalysisFile);
		}
	}
	File.close(gapAnalysisFile);
}

function processFileGap(input, file, gapAnalysisFile) {
	setBatchMode(true);
	
	print("Performing gap analysis on ", file);
	closeROI();
	run("Set Measurements...", "area centroid redirect=None decimal=3");
	run("ROI Manager...");
	run("Clear Results");
	
	open(input + File.separator + file);

	w = getWidth();
	h = getHeight();
	makeRectangle(1, 1, w-2, h-2);
	run("Crop");
	run("Canvas Size...", "width="+w+" height="+h+" position=Center zero");

	run("Max Inscribed Circles", "minimum_disk=" + minimumGapDiameter + " use minimum_similarity=0.50 closeness=5");
	roiManager("Show All");
	roiManager("Set Color", "red");
	roiManager("Set Line Width", 3);
	run("Flatten");
	saveAs("PNG", input + "/GapAnalysis/"+file +"_GapImage.png");
	
	roiManager("Measure");

	nROIs = roiManager("count");

//-----------------------------------------------------------------------
	// Computing average statistics
	if (nROIs > 0)
	{
		totalArea = 0;
		varianceNumerator = 0;
		areaCol = newArray(0);
	  	for (a=0; a<nResults(); a++) {
		    areaCol=append(areaCol,getResult("Area",a));
		    totalArea += getResult("Area",a);
		}
		mean = totalArea/nResults();
	
		for (a=0; a<nResults(); a++) {
			    residual = (mean - getResult("Area",a)) * (mean - getResult("Area",a));
			    varianceNumerator += residual;
			}
		variance = varianceNumerator/(nResults()-1);
		sd = sqrt(variance);
		arr2 = Array.sort(areaCol);
		median = percentile(arr2,0.5);
		percentile5 = percentile(arr2,0.05);
		percentile95 = percentile(arr2,0.95);

		print(gapAnalysisFile,  file + " " + mean + " " + sd + " " + percentile5 + " " + median + " " + percentile95);
	
	//-----------------------------------------------------------------------
		
		for(j=0; j<nROIs;j++){
		roiManager("Select", 0);
		roiManager("Delete");
		}
			
		selectWindow("Results");
		saveAs("Results", input + "/GapAnalysis/IndividualGaps_"+file +".csv");
		
		run("Clear Results");
	}
	close("*");
	setBatchMode(false);
	
}

function append(arr, value) {
     arr2 = newArray(arr.length+1);
     for (i=0; i<arr.length; i++)
        arr2[i] = arr[i];
     arr2[arr.length] = value;
     return arr2;
  }

function percentile(arr,p) { 
	// comes from https://stackoverflow.com/questions/2374640/how-do-i-calculate-percentiles-with-python-numpy
	n = arr.length;
	if(n==0)
	{
		q=-1
	}
	k = (n-1)*p;
	f = floor(k);
	c = -floor(-k); // There is no ceiling function in ijm
	if(f==c)
	{
		q = arr[k];
	}else {
		d0=arr[f] * (c-k);
		d1 = arr[c] * (k-f);
		q = d0+d1;
	}
	return q;
}

function closeROI() {

	if (isOpen("ROI Manager")) {
		 selectWindow("ROI Manager");
		 run("Close");}
}

function processFolderAlignment(input) { // this should be outputMasks folder
	list = getFileList(input);
	list = Array.sort(list);
	list2 = newArray(0);
	anamorfIndex = 0;
	for (j = 0; j < list.length; j++) {
		if(startsWith(list[j], 'AnaMorf')){
			anamorfIndex = j;
		} else {
			list2=append(list2,list[j]);
		}
	}

	alignmentVec = newArray(list2.length);

	for (i = 0; i < list2.length; i++) {
		processFileAlignment(input, list2[i], alignmentVec, i);
	}
	return alignmentVec;
}

function getAlignmentVecOrder(input) { // this should be outputMasks folder
	list = getFileList(input);
	list = Array.sort(list);
	list2 = newArray(0);
	anamorfIndex = 0;
	for (j = 0; j < list.length; j++) {
		if(startsWith(list[j], 'AnaMorf')){
			anamorfIndex = j;
		} else {
			list2=append(list2,list[j]);
		}
	}
	return list2;
}


function processFileAlignment(input,  file, alignmentVec, i) {
	setBatchMode(true);
	open(input + File.separator + file);
	print("Performing alignment analysis on ", file);
	run("Out [-]");
	
	T=getTitle();
	run("OrientationJ Dominant Direction");
	IJ.renameResults("Dominant Direction of "+T,"Results");
	alignmentVec[i]=getResult("Coherency [%]",0);

	run("Clear Results");
	close("*");
	setBatchMode(false);
}

function processFolderDimensions(input){
	list = getFileList(input);
	list = Array.sort(list);
	list2 = newArray(0);
	anamorfIndex = 0;
	for (j = 0; j < list.length; j++) {
		if(startsWith(list[j], 'AnaMorf')){
			anamorfIndex = j;
		} else {
			list2=append(list2,list[j]);
		}
	}

	dimensionVec = newArray(list2.length);

	for (i = 0; i < list2.length; i++) {
		file = list2[i];
		setBatchMode(true);
		open(input + File.separator + file);
		h=getHeight();
		w=getWidth();
		dimensionVec[i]=h*w;
		close("*");
		setBatchMode(false);
	}
	return dimensionVec;
}

function tidyResults(outputHDM, inputAnamorf, inputEligible,alignmentVec){
	hdmResultsFile = outputHDM + File.separator + HDM_RESULTS_FILENAME;
	hdmResults = split(File.openAsString(hdmResultsFile), "\n");
	anaMorfFiles = getFileList(inputAnamorf);
	anaMorfFolderIndex = -1;
	
	for(i=0; i < anaMorfFiles.length; i++){
		if(checkIsAnaMorf(inputAnamorf, anaMorfFiles[i])){
			anaMorfFolderIndex = i;
			break;
		}
	}
	if(anaMorfFolderIndex > -1){
		anaMorfResults = newArray(0);
		for(j=0; j < anaMorfFiles.length; j++){
			if(checkIsAnaMorf(inputAnamorf, anaMorfFiles[j])){
				anaMorfResultsFile = inputAnamorf + File.separator + anaMorfFiles[j] + File.separator + ANAMORF_RESULTS_FILENAME;
				currentAnaMorfResults = split(File.openAsString(anaMorfResultsFile), "\n");
				if(anaMorfResults.length < 1){
					anaMorfResults = currentAnaMorfResults;
				} else{
					for(k=0; k<anaMorfResults.length;k++){
						anaMorfLine = split(currentAnaMorfResults[k], ",");
						anaMorfResults[k] = anaMorfResults[k] + "," + anaMorfLine[1];
					}
				}
			}
		}
		baseOutputFilepath = File.getParent(outputHDM) + File.separator + TWOMBLI_RESULTS_FILENAME;
		outputFilepath = baseOutputFilepath + ".csv";

		if (File.exists(outputFilepath)) {
		    print("\nOverwriting " + outputFilepath);
		}

		
		twombliResultsFile = File.open(outputFilepath);
		for(i = 0; i < anaMorfResults.length; i++){
			hdmIndex = i;
			if(i==1){
				i++;
			}
			if(i==0){
				hdm = "% High Density Matrix";
				alignment = "Alignment";
				dimension = "TotalImageArea";
			} else {
				anaMorfLine = split(anaMorfResults[i], ",");
				hdmIndex =  matchHDMResult(anaMorfLine[0], hdmResults);

				hdmLine = split(hdmResults[hdmIndex], ",");
				hdm = hdmLine[hdmLine.length - 1];
				hdm = 1-hdm;

				alignmentIndex = matchAlignmentResult(anaMorfLine[0]);
				alignment = alignmentVec[alignmentIndex];
				dimension = dimensionsVec[alignmentIndex];
			}

			
			line = anaMorfResults[i] + "," + hdm + "," + alignment + "," + dimension;
			print(twombliResultsFile, line);
		}
		File.close(twombliResultsFile);
	}
}

function checkIsAnaMorf(inputDir, inputFile){
	if(File.isDirectory(inputDir + File.separator + inputFile) && startsWith(inputFile, "AnaMorf")){
		return true;
	} else {
		return false;
	}
}

function matchHDMResult(imageName, hdmResults){
	for(i = 0; i < hdmResults.length; i++){
		line = hdmResults[i];
		splitLine = split(line, ",");
		if(startsWith(imageName, splitLine[1])){
			return i;
		}
	}
	return -1;
}

function matchAlignmentResult(imageName){
	for(i = 0; i < alignmentVecOrder.length; i++){
		if(startsWith(imageName, alignmentVecOrder[i])){
			return i;
		}
	}
	return -1;
}

function numFiles(dir) {
	count=0;
    list = getFileList(dir);
    for (i=0; i<list.length; i++) {
        if (endsWith(list[i], "/"))
            numFiles(""+dir+list[i]);
        else
            count++;
    }
    return count;

}


function emptyFolder(input){
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i])) {
		    emptyFolder(input + File.separator + list[i]);
		    dummy_var = File.delete(input + File.separator + list[i]);
		} else {
		    dummy_var = File.delete(input + File.separator + list[i]);
		}
	}
}

function setLogWindow(ratio, font_size) {
	h = screenHeight;
	w = screenWidth;
	h_sz = h*ratio;
	w_sz = w*ratio;

	script =
	    "importClass(Packages.ij.WindowManager);\n"+
	    "lw = WindowManager.getWindow('Log');\n"+
	    "if (lw!=null) {\n"+
	    "   tp = lw.getTextPanel().setFont(new Font(\"SanSerif\", Font.PLAIN, " + font_size + "), true);\n"+
	    "   lw.setLocation(0,0);\n"+                            
	    "   lw.setSize(" + w_sz + ", " + h_sz + ");\n"+                            
	    "}\n";
	// print(script);
	eval("script", script);
}