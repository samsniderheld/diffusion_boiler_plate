<?php
// Cria um diretório temporário para armazenar os arquivos .txt
$dir = 'temp_files_' . uniqid();
mkdir($dir);

// Cria múltiplos arquivos .txt com conteúdo de exemplo
file_put_contents("file1.txt", "Conteúdo do arquivo 1");
file_put_contents("file2.txt", "Conteúdo do arquivo 2");
file_put_contents("file3.txt", "Conteúdo do arquivo 3");

// Cria um arquivo zip e adiciona os arquivos .txt
$zip = new ZipArchive;
$zipname = 'files_' . uniqid() . '.zip';
if ($zip->open($zipname, ZipArchive::CREATE) === TRUE) {
    $files = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($dir));
    foreach ($files as $file) {
        if ($file->isDir()){
            continue;
        }
        $filePath = $file->getRealPath();
        $relativePath = substr($filePath, strlen($dir) + 1);
        $zip->addFile($filePath, $relativePath);
    }
    $zip->close();

    // Configura os cabeçalhos apropriados para o download do arquivo zip
    header('Content-Type: application/zip');
    header('Content-Disposition: attachment; filename="' . $zipname . '"');
    header('Content-Length: ' . filesize($zipname));

    // Envia o arquivo zip para download
    readfile($zipname);
    
    // Limpa os arquivos temporários
    array_map('unlink', glob("*"));
    rmdir($dir);
    unlink($zipname);
} else {
    echo "Falha ao criar arquivo zip";
}
?>