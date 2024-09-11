// JavaScript
const fileInput = document.getElementById('fileInput');
const drop_area = document.getElementById('drop_area');
const images = new Array();
let CURRENT_INDEX_IMAGE,
    CONTROLLERS = {},
    TERMS,
    PREFIX_CAPITION = "",
    SUFIX_CAPITION = "",
    TXT_FILE_STRING ="",
    CAPTION_FILE=[],
    labelTXT;

// Prevent default behavior for drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  drop_area.addEventListener(eventName, preventDefaults, false);
  document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Highlight drop area when a file is being dragged over it
['dragenter', 'dragover'].forEach(eventName => {
  drop_area.addEventListener(eventName, highlight, false);
});
['dragleave', 'drop'].forEach(eventName => {
  drop_area.addEventListener(eventName, unhighlight, false);
});

function highlight() {
  drop_area.classList.add('highlight');
}

function unhighlight() {
  drop_area.classList.remove('highlight');
}

// Handle dropped files
drop_area.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;

  handleFiles(files);
}

// Handle files once they are dropped
function handleFiles(files) {
  [...files].forEach(previewFile);

  $(".dg.ac").fadeIn("medium");
}

// Listen for file selection from the input field
fileInput.addEventListener('change', e => {
  handleFiles(e.target.files);
});

// Preview the file
function previewFile(file) {
    const reader = new FileReader();

    $("#drop_area").css({
        "left":"0",
        "margin-left":"0"
    });

    $("#thumb_droped_img").css("opacity","1");

    var _currentImage = images.push({file_obj:file});

    $("#thumbnails").append("<img id='"+String(_currentImage-1)+"'>");

    reader.addEventListener("load", e => {loadImages(e, _currentImage-1)});

    reader.readAsDataURL(file);
}

function loadImages(e,indexImage){
    images[indexImage].src = e.target.result;
    images[indexImage].caption = {};
    images[indexImage].captionSet = false;
    $("#thumb_droped_img").append("<img index='"+indexImage+"' name='"+images[indexImage].file_obj.name+"' src='"+e.target.result+"'>");
    $("#all_preview_wrapper").append("<img index='"+indexImage+"' name='"+images[indexImage].file_obj.name+"' src='"+e.target.result+"'>");
    $("#"+indexImage).attr({
        "onclick": "clickChangeImage(this)",
        "name": images[indexImage].file_obj.name,
        "src": e.target.result
    });
    // $("#thumbnails").append("<img onclick='clickChangeImage(this)' id='"+indexImage+"' name='"+images[indexImage].file_obj.name+"' src='"+e.target.result+"'>");
}

function refreshImageBorders(){
    for(var i=0; i<images.length; i++){
        if(images[i].captionSet){
            $("#"+i).addClass("captioned");
        }
    }
}

function clickChangeImage(e){
    refreshImageBorders();

    $("#current_img").attr("src",images[e.id].src);
    CURRENT_INDEX_IMAGE = e.id;

    if(images[CURRENT_INDEX_IMAGE].captionSet){
        refreshGUI();
    } else{
        resetGUI();
    }
}

function resetGUI(){
    for(var i=0; i<TERMS.length; i++){
        CONTROLLERS[TERMS[i]].setValue(false);
    }
}

function refreshGUI(){
    for(var i=0; i<TERMS.length; i++){
        CONTROLLERS[TERMS[i]].setValue(images[CURRENT_INDEX_IMAGE].caption[TERMS[i]]);
    }
}

//dat.gui
var guiObj = {
    inputLabels: finishUploadStep,
    finishLabels: finishLabelsStep,
    next:nextImage,
    prev:prevImage,
    prefix:"",
    sufix:"very fine details, 8k detail, hq",
    download:generateTxtFiles,
    labelText:"",
    labelButton:addLabel,
}

var gui = new dat.gui.GUI({hideable: false}),
    inputLabels = gui.add(guiObj, 'inputLabels').name('Submit Images'),
    finishLabels;


function finishUploadStep(){
    $(".ac").css("top","147px");

    $("#drag_drop").fadeOut("medium",function(){
        $("#attr_list").fadeIn("medium",function(){
            gui.remove(inputLabels);
            finishLabels = gui.add(guiObj, 'finishLabels').name('Process Labels');
        });
        $("html, body").css("background-color","#343846");
    });
}

function finishLabelsStep(){
    var caption_string = document.getElementById("labels_txt").value;

    if(caption_string.indexOf(",") != -1){
        TERMS = caption_string.split(",");
        labelFolder = gui.addFolder("Labels");
        labelFolder.open();

        for(var i=0; i<TERMS.length; i++){
            guiObj[TERMS[i]] = false;

            if(TERMS[i]==""){
                alert("Error: There is a empty label.");
                gui.removeFolder(labelFolder);

                return false;
            }

            for(var j=0; j<images.length; j++){
                images[j].caption[TERMS[i]] = false;
            }

            CONTROLLERS[TERMS[i]] = labelFolder.add(guiObj, TERMS[i]).onFinishChange(function(value){
                images[CURRENT_INDEX_IMAGE].caption[this.property] = value;

                if(value == true){
                    images[CURRENT_INDEX_IMAGE].captionSet = true
                    this.domElement.parentNode.parentNode.classList.add("selected");
                } else{
                    this.domElement.parentNode.parentNode.classList.remove("selected");
                }

                // (value == true) ? images[CURRENT_INDEX_IMAGE].captionSet = true : images[CURRENT_INDEX_IMAGE].captionSet;
            });
        }

        $("#main_imag").append("<img id='current_img' src='"+images[0].src+"'>");
        CURRENT_INDEX_IMAGE = 0;
        
        gui.remove(finishLabels);
        $("#attr_list").fadeOut("medium", function(){
            $("#img_capt").fadeIn("medium");
            next = gui.add(guiObj, 'next').name('Next Image →');
            prev = gui.add(guiObj, 'prev').name('← Previus Image');
            prefix = gui.add(guiObj, 'prefix').name('Caption Prefix');
            sufix = gui.add(guiObj, 'sufix').name('Caption Sufix');
            download = gui.add(guiObj, 'download').name('Download Captions');

            labelAdditionFolder = gui.addFolder("Add Labels");
            labelTXT = labelAdditionFolder.add(guiObj, 'labelText').name('New Label Text:');
            labelAdditionFolder.add(guiObj, 'labelButton').name('Add New Label');
        });
        $(".ac").css("top","0");
    } else{
        alert("Please input your labels");
    }
}

function addLabel(){
    TERMS.push(guiObj.labelText);
    guiObj[TERMS[TERMS.length-1]] = false;
    
    CONTROLLERS[TERMS[TERMS.length-1]] = labelFolder.add(guiObj, TERMS[TERMS.length-1]).onFinishChange(function(value){
        images[CURRENT_INDEX_IMAGE].caption[this.property] = value;

        if(value == true){
            images[CURRENT_INDEX_IMAGE].captionSet = true
            this.domElement.parentNode.parentNode.classList.add("selected");
        } else{
            this.domElement.parentNode.parentNode.classList.remove("selected");
        }

    });

    labelTXT.setValue("");
}

function prevImage(){
    refreshImageBorders();

    (CURRENT_INDEX_IMAGE > 0) ? CURRENT_INDEX_IMAGE-- : CURRENT_INDEX_IMAGE;
    $("#current_img").attr("src",images[CURRENT_INDEX_IMAGE].src);

    if(images[CURRENT_INDEX_IMAGE].captionSet){
        refreshGUI();
    } else{
        resetGUI();
    }
};
function nextImage(){
    refreshImageBorders();

    (CURRENT_INDEX_IMAGE < images.length-1) ? CURRENT_INDEX_IMAGE++ : CURRENT_INDEX_IMAGE;
    $("#current_img").attr("src",images[CURRENT_INDEX_IMAGE].src);

    if(images[CURRENT_INDEX_IMAGE].captionSet){
        refreshGUI();
    } else{
        resetGUI();
    }
};

function generateTxtFiles(){
    PREFIX_CAPITION = guiObj.prefix;
    SUFIX_CAPITION = guiObj.sufix;

    for(var i=0; i<images.length; i++){
        if(images[i].captionSet){
            images[i].captionTxt = PREFIX_CAPITION;
            for(var j=0; j<TERMS.length; j++){
                if(images[i].caption[TERMS[j]]){
                    images[i].captionTxt += TERMS[j]+",";
                }
            }
            images[i].captionTxt += SUFIX_CAPITION;

            CAPTION_FILE.push({"file_name":images[i].file_obj.name,"prompt":images[i].captionTxt})

            var _imgName = images[i].file_obj.name.split(".")
            images[i].fileName = _imgName[0];
        }
    }

    // fetch('generateFiles.php', {
    //     method: 'POST',
    //     headers: {
    //         'Content-Type': 'application/json'
    //     },
    //     body: JSON.stringify(CAPTION_FILE)
    // })
    // .then(response => response.blob())
    // .then(blob => {
    //     const downloadUrl = window.URL.createObjectURL(blob);
    //     const a = document.createElement('a');
    //     a.href = downloadUrl;
    //     a.download = 'captions.jsonl';
    //     document.body.appendChild(a);
    //     a.click();
    //     window.URL.revokeObjectURL(downloadUrl);
    //     document.body.removeChild(a);
    // })
    // .catch(error => {
    //     console.error('Erro:', error);
    // });
    
    // Create JSONL content
    const jsonlContent = CAPTION_FILE.map(item => JSON.stringify(item)).join('\n');
    
    // Create a blob from the JSONL content
    const blob = new Blob([jsonlContent], { type: 'text/plain' });
    
    // Create a download link
    const downloadLink = document.createElement('a');
    downloadLink.download = 'caption.jsonl';
    downloadLink.href = URL.createObjectURL(blob);
    downloadLink.style.display = "none";
    
    // Add the download link to the document
    document.body.appendChild(downloadLink);
    
    // Trigger the download
    downloadLink.click();

    CAPTION_FILE = [];
}