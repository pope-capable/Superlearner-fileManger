const jwt = require('jsonwebtoken')
const Util = require('./Utils');

const util = new Util();


function MSP(req, res, next) {
    if(!req.headers.token){
        util.setError(400, 'Please provide authorization token')
        return util.send(res);
    }else{
        jwt.verify(req.headers.token, 'Serendipity?2050Rocks!', (err, valid) => {
            if(err){
              util.setError(401, 'Invalid token')
              return util.send(res);
            }
            else{
                next();
            }
        })
    }

}

module.exports = MSP;
