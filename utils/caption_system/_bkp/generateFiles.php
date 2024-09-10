<?php
// Get the JSON data from the POST request
$inputJSON = file_get_contents('php://input');
$input = json_decode($inputJSON, TRUE); // Convert the JSON data to a PHP array

// echo "return - ".$input[0]['captionTxt'];
// echo "return - ".$input[0]['fileName'];
// echo $input[0]['fileName'];

$fileContents = array();

for ($i = 0; $i < count($input); $i++) {
    if(isset($input[$i]['captionSet'])){
        // echo $input[$i]['fileName'].'.txt';
        // echo $input[$i]['captionTxt'];

        $fileContents += array( $input[$i]['fileName'].'.txt' => $input[$i]['captionTxt']);
    }
}

// Create a zip archive
$zip = new ZipArchive;
$zipName = 'files.zip';

if ($zip->open($zipName, ZipArchive::CREATE) === TRUE) {
    // Add each text file to the zip archive
    foreach ($fileContents as $fileName => $content) {
        $zip->addFromString($fileName, $content);
    }
    $zip->close();

    header('Content-Type: application/zip');
    header('Content-Disposition: attachment; filename="' . $directory . '.zip"');
    header('Content-Length: ' . filesize($zipName));
    readfile($zipName);
    // Clean up - delete the temporary directory and the zip file
    array_map('unlink', glob("$directory/*"));
    rmdir($directory);
    unlink($zipName);
}
?>