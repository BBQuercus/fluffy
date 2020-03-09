// Cookies

// Makes form selection (selector) active based on cookie (cname)
function addSelectionActive(selector, cookie) {
    const imageSelector = document.getElementById(selector);
    const imageSelection = getCookie(cookie);
    for (let i = 0; i < imageSelector.childNodes.length; i++) {
        if (imageSelector.childNodes[i].value === imageSelection) {
            imageSelector.childNodes[i].setAttribute('selected', 'selected');
        }
    }
}

// Returns cookie value based on name (cname)
function getCookie(cookie) {
    let name = cookie + "=";
    let decodedCookie = decodeURIComponent(document.cookie);
    let ca = decodedCookie.split(';');
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            c = c.substring(name.length, c.length);
            return c.replace(/['"]+/g, '');
        }
    }
    return "";
}
