const Util = require('./Utils');

const util = new Util();


function SE(req, res, next) {
    if(!req.headers.secret_key){
        util.setError(400, 'Please provide secret_key')
        return util.send(res);
    }else{
            if(req.headers.secret_key != "99.99%_accuracy"){
              util.setError(401, 'Invalid Key')
              return util.send(res);
            }
            else{
                next();
            }
    }

}

module.exports = SE;