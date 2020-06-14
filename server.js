import express from 'express';
import db from './app/utils/db';
import middlewares from './app/routes/middleware';
import routes from './app/routes';

const app = express();
/* Serve static files */
app.use('/storage',express.static(process.env.STORAGE_PATH));
middlewares(app);
routes(app);
(async()=>{
    try{
        await db.sync();
        app.listen(process.env.PORT,(err)=>{
            if(err){
                console.log("Server connection failed");
                throw err;
            }

            console.log("Connection established on port "+process.env.PORT);
        });
    } catch(err){
        console.log("Database connection error");
        throw err;
    }
})();