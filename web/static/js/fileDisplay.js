// File upload name display
function fileUploaderHTML(event) {
    const fileUploader = document.getElementById('file-uploader');
    const fileUploaderLabel = document.getElementById('file-uploader-label');
    if (fileUploader.files.length != 0) {
        fileUploaderLabel.innerHTML = fileUploader.value.split('\\').slice(-1)[0];
    }
}

// File upload and list
function makeDelButton(number) {
    let btn = document.createElement('button');
    btn.id = `btn-del-${number}`;
    btn.classList.add('btn');
    btn.classList.add('btn-outline-danger');
    let icon = document.createElement('img');
    icon.src = '/static/images/trash.svg';
    // <img src="/assets/img/bootstrap.svg" alt="" width="32" height="32" title="Bootstrap"></img>
    // icon.className = 'material-icons text-muted md-24';
    // icon.innerText = 'delete';
    btn.appendChild(icon)
    return btn
}

// Adds rows to table based on file
function updateFileList(filename, number, table) {
    const tr = document.createElement('tr');
    tr.id = `row-file-${number}`

    // const thNumber = docume~nt.createElement('th');
    const thOption = document.createElement('td');
    const thFileName = document.createElement('td');
    const delButton = makeDelButton(number)

    // thNumber.appendChild(document.createTextNode(number));
    thFileName.appendChild(document.createTextNode(filename.name));
    thOption.appendChild(delButton);

    // tr.appendChild(thNumber);
    tr.appendChild(thOption);
    tr.appendChild(thFileName);
    table.appendChild(tr);

    return delButton;
}

// Creates a new download button (no functionality)
function makeDownloadButton(number) {
    let btn = document.createElement('button');
    btn.id = `btn-download-${number}`;
    btn.classList.add('btn');
    btn.classList.add('btn-outline-primary');
    btn.classList.add('btn-download');
    let icon = document.createElement('img');
    icon.src = '/static/images/download.svg';
    btn.appendChild(icon)
    return btn
}

// Deletes "old" download buttons
function removeDownloadButtons(fileList) {
    if (document.querySelector('#btn-download-all')) {
        for (let key in fileList) {
            document.querySelector(`#btn-download-${key}`).remove();
        }
        document.querySelector('#btn-download-all').remove();
    }
}