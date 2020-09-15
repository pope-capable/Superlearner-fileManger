const Util = require('./Utils');
var multer  = require('multer')
var multerS3 = require('multer-s3')
var aws = require('aws-sdk')

aws.config.update({
    secretAccessKey: 'leJcn4UlL2mVXbSHXA1Ka+f/UeF1E9JLOccYOV7J',
    accessKeyId: 'AKIAJM4B2RBJFKJIXQAA',
});

var s3 = new aws.S3()


const util = new Util();


function uploadFile(req, res, next) {
    var upload = multer({
        storage: multerS3({
        s3: s3,
        bucket: 'superlearner',
        metadata: function (req, file, cb) {
          cb(null, {fieldName: file.fieldname});
        },
        key: function (req, file, cb) {
          cb(null, Date.now().toString())
          var getdate = Date.now().toString()
          var useKey = `${getdate + file.originalname}`
          cb(null, useKey); //use Date.now() for unique file keys
    
        }
      })
     })
    next()
}

module.exports = uploadFile;