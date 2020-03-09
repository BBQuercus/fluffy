// Utilities

// Returns length of object - Object.size(nameOfObject)
Object.size = function (obj) {
    var size = 0, key;
    for (key in obj) {
        if (obj.hasOwnProperty(key)) size++;
    }
    return size;
};

// Automatically downloads file from url
function download(dataurl) {
    let url = `${window.location.origin}/download/${dataurl}`
    let a = document.createElement('a');
    a.href = url;
    a.setAttribute('download', true);
    console.log(a);
    a.click();
}

// Show success/failure message - make sure to $.hide(); before
function alertMessage(id, time) {
    $(`#${id}`).fadeTo(time, 500).slideUp(500, function () {
        $(`#${id}`).alert('close');
    });
}

// Delete specific file on the server side
function deleteFileOnServer(filename) {
    let xhr = new XMLHttpRequest();
    xhr.open('GET', `/delete/${filename}`);
    xhr.send();
}

// Activates current page in navigation bar
function activateNavItem(pathname) {
    let navItem = document.getElementById(`nav-item-${pathname}`);
    if (navItem) {
        navItem.classList.add('active');
    }
}
