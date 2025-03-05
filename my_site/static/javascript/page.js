const uploadbtn = document.getElementById('upload-button')
const uploadform = document.getElementById('upload-form')
const iframe = document.getElementById('iframe')
const uploaded = document.getElementById('uploaded-dataset-div')

const dataset = document.getElementById('dataset')
const trainbtn = document.getElementById('train-btn')
const removebtn = document.getElementById('remove-btn')

/*

uploadbtn.addEventListener('click', (e) => {
    e.preventDefault()
    
    const payload = new FormData(uploadform)

    fetch(`${window.location.origin}/getHTML`, {
        method: 'POST',
        body: payload,
    })
    .then(res => {
        if(res.status != '200') {
            return res.json().then(data => {
                alert(data.message || 'Error');
                throw new Error(data.message || 'Error');
            });
        } else {
            return res.text()
        }
    })
    .then(data => {
        const iframedoc = iframe.contentDocument || iframe.contentWindow.document
        iframedoc.open()
        iframedoc.write(data)
        iframedoc.close()

        //uploadbtn.disabled = true
        uploaded.scrollIntoView({behavior: 'smooth'})
    })
    .catch(err => {})
})

trainbtn.addEventListener('click', (e) => {
    e.preventDefault()

    uploadform.action = `${window.location.origin}/train`
    uploadform.method = 'POST'

    uploadform.submit()
})

*/