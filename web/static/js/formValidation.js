// Validation
function validateSingle() {
    const btnPredict = document.querySelector('#btn-predict');
    btnPredict.setAttribute('disabled', 'true');

    if (document.querySelector('#file-uploader').files.length === 0) return;
    if (document.querySelector('#image-type').value == '') return;
    if (document.querySelector('#model-type').value == '') return;

    enableButton(btnPredict);
}

function validateBatch() {
    const btnPredict = document.querySelector('#btn-predict');
    btnPredict.setAttribute('disabled', 'true');

    if (document.querySelector('#image-type').value == '') return;
    if (document.querySelector('#model-type').value == '') return;

    enableButton(btnPredict);
}


// "Turns on" button with bootstrap classes
function enableButton(button) {
    button.removeAttribute('disabled');
    button.classList.remove('btn-light');
    button.classList.add('btn-primary');
}